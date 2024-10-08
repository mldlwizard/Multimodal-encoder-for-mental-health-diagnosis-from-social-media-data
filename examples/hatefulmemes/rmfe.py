# export PYTHONPATH=$PYTHONPATH:`pwd`
#------------------------- Import Libraries -----------------------------------#

import os
import torch
import pandas as pd
from torch import nn

from dataset.dataloader import HatefulMemesDataset
from preprocessing.embeddings import Embeddings
from preprocessing.fusions import *
from models.basic_models import MLP
# from config.config import configuration
from supervised.train import *
from supervised.plots import *
from loss_functions.loss_functions import *

#-------------------- Initialize Parameters ----------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
home_path = os.getcwd()
os.chdir("/work/socialmedia/multimodal_encoder/RAW_DATA/")

#------------------- Dataloader -------------------#

train_dataset = HatefulMemesDataset(path= configuration['Dataset']['path_to_data'], 
                                    dataloader_type= 'train', 
                                    batch_size= configuration['Dataset']['batch_size'], 
                                    shuffle=configuration['Dataset']['shuffle'],
                                    cache_size= configuration['Dataset']['cache_size'],
                                    data_filepath= None)


val_dataset = HatefulMemesDataset(path= configuration['Dataset']['path_to_data'], 
                                    dataloader_type= 'validation', 
                                    batch_size= configuration['Dataset']['batch_size'], 
                                    shuffle=configuration['Dataset']['shuffle'],
                                    cache_size= configuration['Dataset']['cache_size'],
                                    data_filepath="hatefulmemes/dev.jsonl")

test_dataset = HatefulMemesDataset(path= configuration['Dataset']['path_to_data'], 
                                    dataloader_type= 'test', 
                                    batch_size= configuration['Dataset']['batch_size'], 
                                    shuffle=configuration['Dataset']['shuffle'],
                                    cache_size= configuration['Dataset']['cache_size'],
                                    data_filepath="hatefulmemes/test.jsonl")

#---------------- Model Definition --------------------#

class CombinedModel(nn.Module):
    def __init__(self, modality1, modality2, configuration,device):
        super().__init__()
        self.modality1 = modality1
        self.modality2 = modality2
        self.config = configuration
        self.device = device

    def forward(self, input1, input2):
        image_output = self.modality1(**input1)['last_hidden_state'].to(self.device)
        image_output = torch.mean(image_output,1).to(self.device)
        
        text_output = self.modality2(**input2)['last_hidden_state'].to(self.device)
        text_output = torch.mean(text_output,1).to(self.device)

        fusion_output = Concat().to(self.device)([image_output, text_output]).to(self.device)

        head_output = MLP(in_channels=fusion_output.shape[1],
                            num_classes=self.config['Models']['mlp_num_classes'],
                            hidden_sizes=self.config['Models']['mlp_hidden_sizes'], 
                            dropout_probability= self.config['Models']['mlp_dropout_prob']).to(self.device)(fusion_output).to(self.device)
        
        return head_output


# Image Models
img_processor = configuration['Models']['image_processor']
model_img = configuration['Models']['image_model'].to(device)

# Text Models
tokenizer_txt = configuration['Models']['text_tokenizer']
model_txt = configuration['Models']['text_model'].to(device)

model = CombinedModel(modality1 = model_img,modality2 = model_txt,configuration = configuration,device = device).to(device)

criterion = RMFE_object().to(device)
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])

model_head, train_metrics = train(train_dataset, val_dataset, configuration, device, model, criterion, optimizer)
print('Finished Training')

# test_metrics = test(model_head, test_dataset, configuration, device)
# print('Finished Testing')

train_metrics = pd.DataFrame(train_metrics)
# test_metrics = pd.DataFrame(test_metrics)

print("Metrics: ")
print(train_metrics)
# print(test_metrics)

# Plots
os.chdir(home_path)
plots(train_metrics, f'hatefulmemes_{idx}')





























