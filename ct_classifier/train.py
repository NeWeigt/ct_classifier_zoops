'''
    Training script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.

    2022 Benjamin Kellenberger
'''

from csv import writer
from csv import writer
import os
import argparse
import yaml
import glob
from tqdm import trange

import torch # this imports pytorch
import torch.nn as nn # this contains our loss function 
from torch.utils.data import DataLoader # the pytorch dataloader class will take care of all kind of parallelization during training
from torch.optim import SGD # this imports the optimizer
import torchvision # this contains some helper functions for vision-related tasks
import matplotlib.pyplot as plt

# let's import our own classes and functions!
from util import init_seed
from dataset import CTDataset
from model import CustomResNet18
from torch.utils.tensorboard import SummaryWriter
import json



def create_dataloader(cfg, split='train'):
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''
    dataset_instance = CTDataset(cfg, split)        # create an object instance of our CTDataset class

    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=cfg['batch_size'],
            shuffle=split=='train',
            num_workers=cfg['num_workers']
        )
    return dataLoader



def load_model(cfg):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    model_instance = CustomResNet18(cfg['num_classes'])         # create an object instance of our CustomResNet18 class

    # load latest model state
    model_chkpt_path = cfg['model_checkpoint_path']
    if model_chkpt_path is None or model_chkpt_path == '':
        print('Starting new model (no checkpoint path specified)')
        start_epoch = 0
    elif os.path.exists(model_chkpt_path):
        print(f'Loading model state from {model_chkpt_path}...')
        checkpoint = torch.load(open(model_chkpt_path, 'rb'))
        model_instance.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint.get('epoch', 0)
    else:
        # no save state found; start anew
        print(f'WARNING NELE: No model state found at {model_chkpt_path}.')
        print('Starting new model')
        start_epoch = 0

    return model_instance, start_epoch



def save_model(cfg, epoch, model, stats):
    # make sure save directory exists; create if not
    os.makedirs('model_states', exist_ok=True)

    # get model parameters and add to stats...
    stats['model'] = model.state_dict()

    # ...and save
    torch.save(stats, open(f'model_states/{epoch}.pt', 'wb'))
    
    # also save config file if not present
    cfpath = 'model_states/config.yaml'
    if not os.path.exists(cfpath):
        with open(cfpath, 'w') as f:
            yaml.dump(cfg, f)

            

def setup_optimizer(cfg, model):
    '''
        The optimizer is what applies the gradients to the parameters and makes
        the model learn on the dataset.
    '''
    optimizer = SGD(model.parameters(),
                    lr=cfg['learning_rate'],
                    weight_decay=cfg['weight_decay'])
    return optimizer

  
       
def train_val_test(cfg, dataLoader, model,  writer, optimizer=None, split='train', epoch=None):
    '''
     Our actual training function.
    '''
    device = cfg['device']
    # put model on device
    model.to(device)
    if split=='train':
        # put the model into training mode
        # this is required for some layers that behave differently during training
        # and validation (examples: Batch Normalization, Dropout, etc.)
        model.train()
    else:
        model.eval()
    # loss function
    #  note: if you're doing multi target classification, use nn.BCEWithLogitsLoss() and convert labels to float
    criterion = nn.CrossEntropyLoss()
    # running averages
    losses=[]
    accuracies=[]
    #all_gt_labels=[]
    #all_pred_labels=[]
    all_predictions = []
    all_labels = []
    all_image_names = []
    all_confidences = []
    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    for idx, (data, labels, image_names) in enumerate(dataLoader):
       # see the last line of file "dataset.py" where we return the image tensor (data) and label
       # put data and labels on device
        data, labels = data.to(device), labels.to(device)
        
        # forward pass
        prediction = model(data)
        if split == "train":
            # reset gradients to zero
            optimizer.zero_grad()
        # loss
        loss = criterion(prediction, labels)
        if split == 'train':
            # backward pass (calculate gradients of current batch)
            loss.backward()
            # apply gradients to model parameters
            optimizer.step()
            # log statistics
            # loss_total += loss.item()   
        losses.append(loss.item())                      # the .item() command retrieves the value of a single-valued tensor, regardless of its data type and device of tensor
        pred_label = torch.argmax(prediction, dim=1)    # the predicted label is the one at position (class index) with highest predicted value
        pred_confidence = torch.max(torch.softmax(prediction, dim=1), dim=1).values
        oa = torch.mean((pred_label == labels).float()) # OA: number of correct predictions divided by batch size (i.e., average/mean)
        # oa_total += oa.item()
        accuracies.append(oa.item())
        #all_gt_labels.append(gt_labels.cpu())
        #all_pred_labels.append(pred_label.cpu())
        all_predictions.extend(pred_label.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_image_names.extend(image_names)
        all_confidences.extend(pred_confidence.cpu().tolist())
        progressBar.set_description(
            '[{}] Loss: {:.2f}; OA: {:.2f}%'.format(
                split.capitalize(),
                (loss.item()),
                100*(oa.item())
            )
        )
        progressBar.update(1)
    # end of epoch; finalize
    progressBar.close()
    loss_total = sum(losses)/len(losses)           # shorthand notation for: loss_total = loss_total / len(dataLoader)
    oa_total = sum(accuracies)/len(accuracies)
    # save all predictions and labels to a json file for further analysis
    results = {
        'predictions': all_predictions,
        'labels': all_labels,
        'image_names': all_image_names,
        'confidences': all_confidences
    }
    
    log_dir = writer.log_dir
    val_dir = os.path.join(log_dir, 'val_predictions')
    os.makedirs(val_dir, exist_ok=True)
    # zero pad epoch number to have files in correct order when sorted by name
    with open(f'{val_dir}/val_predictions_epoch_{epoch:05d}.json', 'w') as f:
        json.dump(results, f)
    return loss_total, oa_total

def main():
    import argparse, yaml, os
    from datetime import datetime
    from torch.utils.tensorboard import SummaryWriter
    # Argument parser
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='configs/model_try.yaml')
    args = parser.parse_args()
    # Load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))
    # Initialize random seed
    init_seed(cfg.get('seed', None))
   
    # TensorBoard writer
    writer = SummaryWriter()
    # Check GPU availability
    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'
    # Initialize dataloaders
    dl_train = create_dataloader(cfg, split='train')
    dl_val = create_dataloader(cfg, split='val')
    
    # Initialize model
    model, current_epoch = load_model(cfg)
    # Setup optimizer
    optim = setup_optimizer(cfg, model)
    # Training loop
    numEpochs = cfg['num_epochs']
    for current_epoch in range(current_epoch, numEpochs):
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')
        loss_train, oa_train = train_val_test(cfg, dl_train, model, writer, optim, split="train", epoch=current_epoch)
        loss_val, oa_val = train_val_test(cfg, dl_val, model, writer, split="val", epoch=current_epoch)
        writer.add_scalar('Loss/train', loss_train, current_epoch)
        writer.add_scalar('Loss/val', loss_val, current_epoch)
        writer.add_scalar('Accuracy/train', oa_train, current_epoch)
        writer.add_scalar('Accuracy/val', oa_val, current_epoch)
        # Combine stats and save
        stats = {
            'loss_train': loss_train,
            'loss_val': loss_val,
            'oa_train': oa_train,
            'oa_val': oa_val
        }
        save_model(cfg, current_epoch, model, stats)
if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    print('Starting training script...')
    main()
