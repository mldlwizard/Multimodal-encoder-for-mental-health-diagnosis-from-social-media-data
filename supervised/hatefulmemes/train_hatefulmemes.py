from supervised.metrics import *
from tqdm import tqdm
import torch
import torch.nn
import torch.nn.functional as F
import numpy as np

def train(train_dataset, val_dataset, configuration, device, model, criterion, optimizer, trial = None):

    ### Preprocessing Models

    # Image processor
    img_processor = configuration['Models']['image_processor']

    # Text tokenizer
    tokenizer_txt = configuration['Models']['text_tokenizer']

    best_val_acc = 0

    train_metrics = {"epoch":[],"num_steps":[],"train_loss":[],"val_loss":[],"train_acc":[],"val_acc":[],"train_precision":[],"val_precision":[],"train_recall":[],"val_recall":[],"train_f1score":[],"val_f1score":[]}
    for epoch in range(1, configuration['Hyperparameters']['epochs']+1):
        num_steps = 0
        val_num_steps = 0
        running_loss = 0.0
        val_running_loss = 0.0
        true = []
        pred = []
        val_true = []
        val_pred = []

        
        # trainit = iter(train_dataset)
        for data in tqdm(range(len(train_dataset)), desc=f"[Epoch {epoch}]",ascii=' >='):
            # data = next(trainit)
            # shuffle the dataset
            
            img_batch = data['img']
            text_batch = data['text']
            labels = torch.from_numpy(data['output']).to(device)
            labels_one_hot = F.one_hot(labels.to(torch.int64).squeeze(), configuration['Models']['mlp_num_classes'])
            
            text_feats = tokenizer_txt(text_batch, return_tensors="pt", padding=True, truncation=True).to(device)
            img_processed = img_processor(img_batch, return_tensors='pt').to(device) 

            outputs = model(img_processed, text_feats).to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            if(configuration['Models']['regularization'] == True):
                # add L2 regularization to the loss function
                regularization_loss = 0
                for param in model.parameters():
                    regularization_loss += torch.sum(torch.square(param))
                
                loss = criterion(outputs, labels_one_hot.to(torch.float32)) + configuration['Loss']['reg_lambda'] * regularization_loss
            else:
                loss = criterion(outputs, labels_one_hot.to(torch.float32))

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            true.extend(labels_one_hot.cpu().detach().numpy())
            pred.extend(outputs.cpu().detach().numpy())
           
            num_steps +=1

        # Validation
        with torch.no_grad():
            valit = iter(val_dataset)
            for j in tqdm(range(len(val_dataset)), desc=f"[Epoch {epoch}]",ascii=' >='):
                val_data = next(valit)

                val_img_batch = val_data['img']
                val_text_batch = val_data['text']

                val_labels = torch.from_numpy(val_data['output']).to(device)
                val_labels_one_hot = F.one_hot(val_labels.to(torch.int64).squeeze(), configuration['Models']['mlp_num_classes'])

                text_feats = tokenizer_txt(val_text_batch, return_tensors="pt", padding=True, truncation=True).to(device)
                img_processed = img_processor(val_img_batch, return_tensors='pt').to(device) 

                val_outputs = model(img_processed, text_feats).to(device)

                val_loss = criterion(val_outputs, val_labels_one_hot.to(torch.float32))
                val_true.extend(val_labels_one_hot.cpu().detach().numpy())
                val_running_loss += val_loss.item()

                val_pred.extend(val_outputs.cpu().detach().numpy())
                
                val_num_steps +=1
        print("Unique ******************* ", np.unique(np.argmax(true, axis=1)))
        train_acc = accuracy(np.argmax(true, axis=1), np.argmax(pred, axis=1))
        val_acc = accuracy(np.argmax(val_true, axis=1), np.argmax(val_pred, axis=1))

        train_precision = precision(np.argmax(true, axis=1), np.argmax(pred, axis=1))
        val_precision = precision(np.argmax(val_true, axis=1), np.argmax(val_pred, axis=1))

        train_recall = recall(np.argmax(true, axis=1), np.argmax(pred, axis=1))
        val_recall = recall(np.argmax(val_true, axis=1), np.argmax(val_pred, axis=1))

        train_f1 = f1score(np.argmax(true, axis=1), np.argmax(pred, axis=1))
        val_f1 = f1score(np.argmax(val_true, axis=1), np.argmax(val_pred, axis=1))

        print(f'Num_steps : {num_steps}, train_loss : {running_loss/num_steps:.3f}, val_loss : {val_running_loss/val_num_steps:.3f}, train_acc : {train_acc}, val_acc : {val_acc}, train_f1score : {train_f1}, val_f1score : {val_f1}')

        if(val_acc > best_val_acc):
            best_val_acc = val_acc
            best_model = model

        train_metrics["epoch"].append(epoch)
        train_metrics["num_steps"].append(num_steps)
        train_metrics["train_loss"].append(running_loss/num_steps)
        train_metrics["val_loss"].append(val_running_loss/val_num_steps)
        train_metrics["train_acc"].append(train_acc)
        train_metrics["val_acc"].append(val_acc)
        train_metrics["train_precision"].append(train_precision)
        train_metrics["val_precision"].append(val_precision)
        train_metrics["train_recall"].append(train_recall)
        train_metrics["val_recall"].append(val_recall)
        train_metrics["train_f1score"].append(train_f1)
        train_metrics["val_f1score"].append(val_f1)

        # # Early Stopping trials
        # if(trial):
        #     trial.report(val_acc, epoch)

        #     if trial.should_prune():
        #         raise optuna.exceptions.TrialPruned()

    return best_model, train_metrics



    