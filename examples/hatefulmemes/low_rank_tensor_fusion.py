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

def low_rank_tensor_fusion(configuration, trial=None):

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

            fusion_in_features = self._calculate_in_features_fusion()
            self.fusion = LowRankTensorFusion(input_dims=[fusion_in_features[0], fusion_in_features[1]], output_dim=512,rank=128)

            self.head = MLP(in_channels=self._calculate_in_features_head(),
                                    num_classes=self.config['Models']['mlp_num_classes'],
                                    hidden_sizes=self.config['Models']['mlp_hidden_sizes'], 
                                    dropout_probability= self.config['Models']['mlp_dropout_prob'])

            if(configuration['Models']['encoder_finetuning'] == False):
                for param in self.modality1.parameters():
                    param.requires_grad = False
                
                for param in self.modality2.parameters():
                    param.requires_grad = False

            for param in self.head.parameters():
                param.requires_grad = True
                
            for param in self.fusion.parameters():
                param.requires_grad = True

        def forward(self, input1, input2):
            image_output = self.modality1(**input1)['last_hidden_state'].to(self.device)
            image_output = torch.mean(image_output,1).to(self.device)
            
            text_output = self.modality2(**input2)['last_hidden_state'].to(self.device)
            text_output = torch.mean(text_output,1).to(self.device)

            fusion_output = self.fusion.to(self.device)([image_output,text_output]).to(self.device)
            # print(self.head.parameters().requires_grad)
            
            head_output = self.head.to(self.device)(fusion_output).to(self.device)

            return head_output

        def _calculate_in_features_fusion(self):
            # Create an example input and pass it through the network to get the output size
            img_batch = torch.randint(0, 255, size=(self.config['Dataset']['batch_size'], 3, 224, 224)).float()
            img_processor = self.config['Models']['image_processor']
            tokenizer_txt = configuration['Models']['text_tokenizer']
            text_batch = "This is a sample input for shape inference"
            text_batch = [text_batch] * self.config['Dataset']['batch_size']
            input1 = img_processor(img_batch, return_tensors='pt').to(self.device)
            input2 = tokenizer_txt(text_batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            # Forward pass until MLP
            image_output = self.modality1(**input1)['last_hidden_state'].to(self.device)
            image_output = torch.mean(image_output,1).to(self.device)
            text_output = self.modality2(**input2)['last_hidden_state'].to(self.device)
            text_output = torch.mean(text_output,1).to(self.device)
            
            return image_output.shape[1],text_output.shape[1]
        
        def _calculate_in_features_head(self):
            # Create an example input and pass it through the network to get the output size
            img_batch = torch.randint(0, 255, size=(self.config['Dataset']['batch_size'], 3, 224, 224)).float()
            img_processor = self.config['Models']['image_processor']
            tokenizer_txt = configuration['Models']['text_tokenizer']
            text_batch = "This is a sample input for shape inference"
            text_batch = [text_batch] * self.config['Dataset']['batch_size']
            input1 = img_processor(img_batch, return_tensors='pt').to(self.device)
            input2 = tokenizer_txt(text_batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            # Forward pass until MLP
            image_output = self.modality1(**input1)['last_hidden_state'].to(self.device)
            image_output = torch.mean(image_output,1).to(self.device)
            text_output = self.modality2(**input2)['last_hidden_state'].to(self.device)
            text_output = torch.mean(text_output,1).to(self.device)
            fusion_output = self.fusion.to(self.device)([image_output,text_output]).to(self.device)
            
            return fusion_output.shape[1]


    # Image Models
    img_processor = configuration['Models']['image_processor']
    model_img = configuration['Models']['image_model'].to(device)

    # Text Models
    tokenizer_txt = configuration['Models']['text_tokenizer']
    model_txt = configuration['Models']['text_model'].to(device)

    model = CombinedModel(modality1 = model_img,modality2 = model_txt,configuration = configuration,device = device).to(device)

    criterion = getattr(nn,configuration['Loss']['loss_fn'])().to(device)
    optimizer = getattr(torch.optim,configuration['Optimizers']['optimizer'])

    optimizer = optimizer([p for p in model.parameters() if p.requires_grad])

    model_head, train_metrics = train(train_dataset, val_dataset, configuration, device, model, criterion, optimizer, trial)
    print('Finished Training')

    return train_metrics

# test_metrics = test(model_head, test_dataset, configuration, device)
# print('Finished Testing')

# train_metrics = pd.DataFrame(train_metrics)
# # test_metrics = pd.DataFrame(test_metrics)

# print("Metrics: ")
# print(train_metrics)
# # print(test_metrics)

# os.chdir(home_path)
# train_metrics.to_csv(home_path + "/results/metrics/" + str(idx) + "/hatefulmemes_low_rank_tensor_fusion_config_" + str(idx) + ".csv",index=False)


# Plots

# plots(train_metrics, f'hatefulmemes_{idx}')





























