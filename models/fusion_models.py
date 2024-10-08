
import torch
from torch import nn
import torch.nn.functional as F

from preprocessing.fusions import *
from preprocessing.embeddings import Embeddings


#------------------------------- Unimodal Img ------------------------------------------#

class UnimodalImg(nn.Module):
    def __init__(self, modality1, modality2, configuration,device):
        super().__init__()
        self.modality1 = modality1
        self.modality2 = modality2
        self.config = configuration
        self.device = device


        self.head = MLP(in_channels=self._calculate_in_features(),
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

    def forward(self, input1, input2):
        image_output = self.modality1(**input1)['last_hidden_state'].to(self.device)
        image_output = torch.mean(image_output,1).to(self.device)
        head_output = self.head.to(self.device)(image_output).to(self.device)
        return head_output
    
    def _calculate_in_features(self):
        # Create an example input and pass it through the network to get the output size
        img_batch = torch.randint(0, 255, size=(self.config['General']['batch_size'], 3, self.config['General']['img_size'], self.config['General']['img_size'])).float()
        img_processor = self.config['Models']['image_processor']
        input1 = img_processor(img_batch, return_tensors='pt').to(self.device) 
        image_output = self.modality1(**input1)['last_hidden_state'].to(self.device)
        image_output = torch.mean(image_output,1).to(self.device)
        return image_output.shape[1]

#------------------------------- Unimodal Txt ------------------------------------------#

class UnimodalTxt(nn.Module):
    def __init__(self, modality1, modality2, configuration,device):
        super().__init__()
        self.modality1 = modality1
        self.modality2 = modality2
        self.config = configuration
        self.device = device

        self.head = MLP(in_channels=self._calculate_in_features(),
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

    def forward(self, input1, input2):

        text_output = self.modality2(**input2)['last_hidden_state'].to(self.device)
        text_output = torch.mean(text_output,1).to(self.device)

        head_output = self.head.to(self.device)(text_output).to(self.device)

        return head_output

    
    def _calculate_in_features(self):
        # Create an example input and pass it through the network to get the output size
        tokenizer_txt = self.config['Models']['text_tokenizer']
        text_batch = "This is a sample input for shape inference"
        text_batch = [text_batch] * self.config['General']['batch_size']
        input2 = tokenizer_txt(text_batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # Forward pass until MLP
        text_output = self.modality2(**input2)['last_hidden_state'].to(self.device)
        text_output = torch.mean(text_output,1).to(self.device)
        
        return text_output.shape[1]



class LateFusion(nn.Module):

    def __init__(self, modality1, modality2, configuration,device):
        super().__init__()
        self.modality1 = modality1
        self.modality2 = modality2
        self.config = configuration
        self.device = device

        self.head = MLP(in_channels=self._calculate_in_features(),
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

    def forward(self, input1, input2):
        image_output = self.modality1(**input1)['last_hidden_state'].to(self.device)
        image_output = torch.mean(image_output,1).to(self.device)
        
        text_output = self.modality2(**input2)['last_hidden_state'].to(self.device)
        text_output = torch.mean(text_output,1).to(self.device)

        fusion_output = Concat().to(self.device)([image_output, text_output]).to(self.device)

        head_output = self.head.to(self.device)(fusion_output).to(self.device)
        
        return head_output

    def _calculate_in_features(self):
        # Create an example input and pass it through the network to get the output size
        img_batch = torch.randint(0, 255, size=(self.config['General']['batch_size'], 3, self.config['General']['img_size'], self.config['General']['img_size'])).float()
        img_processor = self.config['Models']['image_processor']
        tokenizer_txt = self.config['Models']['text_tokenizer']
        text_batch = "This is a sample input for shape inference"
        text_batch = [text_batch] * self.config['General']['batch_size']
        input1 = img_processor(img_batch, return_tensors='pt').to(self.device)
        input2 = tokenizer_txt(text_batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # Forward pass until MLP
        image_output = self.modality1(**input1)['last_hidden_state'].to(self.device)
        image_output = torch.mean(image_output,1).to(self.device)
        text_output = self.modality2(**input2)['last_hidden_state'].to(self.device)
        text_output = torch.mean(text_output,1).to(self.device)
        fusion_output = Concat().to(self.device)([image_output, text_output]).to(self.device)
        
        return fusion_output.shape[1]


#-------------------------------- MLP ------------------------------------------#

class MLP(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_sizes=[128, 64], dropout_probability=[0.5,0.7]):
        super(MLP, self).__init__()
        assert len(hidden_sizes) >= 1 , "specify at least one hidden layer"
        
        self.layers = self.create_layers(in_channels, num_classes, hidden_sizes, dropout_probability)


    def create_layers(self, in_channels, num_classes, hidden_sizes, dropout_probability):
        layers = []
        layer_sizes = [in_channels] + hidden_sizes + [num_classes]
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:
                layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_probability[i]))
            else:
                layers.append(nn.Softmax(dim=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x.view(x.shape[0], -1)
        out = self.layers(out)
        return out