
print(f"\n*********** {configuration['Models']['model_class_name']} ***************\n")

# export PYTHONPATH=$PYTHONPATH:`pwd`
#------------------------- Import Libraries -----------------------------------#

import os
import torch
import pandas as pd
from torch import nn

from dataset.dataloader import load_inference_data
from supervised.train import train, test

#-------------------- Initialize Parameters ----------------------------#

device = configuration['General']['device']
home_path = os.getcwd()

#------------------- Dataloader -------------------#

data_path = configuration['Dataset']['data_path']
dir_root_path = configuration['Dataset']['dir_root_path']
IMG_SIZE = configuration['General']['img_size']


train_dataloader, val_dataloader, test_dataloader, num_classes, _,_,_ = load_inference_data(data_path, 
                                                                                    dir_root_path, 
                                                                                    IMG_SIZE, 
                                                                                    batch_size=configuration['General']['batch_size'])

# Image Models
img_processor = configuration['Models']['image_processor']
model_img = configuration['Models']['image_model'].to(device)

# Text Models
tokenizer_txt = configuration['Models']['text_tokenizer']
model_txt = configuration['Models']['text_model'].to(device)

# Import the module dynamically
module = importlib.import_module("models.fusion_models")

# Get the class object using getattr()
model = getattr(module, configuration['Models']['model_class_name'])
model = model(modality1 = model_img,modality2 = model_txt,configuration = configuration,device = device).to(device)


if(configuration['General']['train']):

    print("\nTraining:")

    try:
        print("Loading Model")
        state_dict = torch.load(configuration['Dataset']['best_model_path'])
        model.load_state_dict(state_dict) 
        print("Model Loaded")
    except:
        pass

    criterion = getattr(nn,configuration['Loss']['loss_fn'])().to(device)
    optimizer = getattr(torch.optim,configuration['Optimizers']['optimizer'])

    optimizer = optimizer([p for p in model.parameters() if p.requires_grad], lr = configuration['Optimizers']['learning_rate'])

    train(train_dataloader, val_dataloader, configuration, device, model, criterion, optimizer)
    
    print('Finished Training')


if(configuration['General']['test']):

    print("\nTesting:")

    state_dict = torch.load(configuration['Dataset']['best_model_path'])
    model.load_state_dict(state_dict) 
    print("Model Loaded")

    test(test_dataloader, configuration, device, model)

    print('Finished Testing')































