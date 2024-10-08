import warnings
warnings.filterwarnings("ignore")

import torch
import pandas as pd
from torch import nn
import yaml
# from config_class import Config
import importlib
# from models.basic_models import MLP
import os
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.trial import TrialState
import time

from dataset.dataloader import HatefulMemesDataset
from preprocessing.embeddings import Embeddings
from preprocessing.fusions import *
from models.basic_models import MLP
# from config.config import configuration
from supervised.train import *
from supervised.plots import plot_loss_accuracy

from examples.hatefulmemes.late_fusion import late_fusion
from examples.hatefulmemes.low_rank_tensor_fusion import low_rank_tensor_fusion
from examples.hatefulmemes.unimodal_img import unimodal_img
from examples.hatefulmemes.unimodal_txt import unimodal_txt

home_path = os.getcwd()

# Load the YAML file
with open(home_path + '/config/config.yaml') as f:
    configuration = yaml.safe_load(f)

base_import_package = importlib.import_module(configuration['Models']['base_import_package'])

configuration['Models']['image_processor'] = getattr(base_import_package,configuration['Models']['image_processor_package'])
configuration['Models']['image_processor'] = configuration['Models']['image_processor'].from_pretrained(configuration['Models']['image_processor_pretrained'])

configuration['Models']['image_model'] = getattr(base_import_package,configuration['Models']['image_model_package'])
configuration['Models']['image_model'] = configuration['Models']['image_model'].from_pretrained(configuration['Models']['image_model_name'])

configuration['Models']['text_tokenizer'] = getattr(base_import_package,configuration['Models']['text_tokenizer_package'])
configuration['Models']['text_tokenizer'] = configuration['Models']['text_tokenizer'].from_pretrained(configuration['Models']['text_processor_pretrained'])

configuration['Models']['text_model'] = getattr(base_import_package,configuration['Models']['text_model_package'])
configuration['Models']['text_model'] = configuration['Models']['text_model'].from_pretrained(configuration['Models']['text_model_name'])


idx=0
directory_path = home_path + "/results/metrics/{}".format(idx+1)

if not os.path.exists(directory_path):
    os.makedirs(directory_path)


# print("------------- UNIMODAL IMG -------------")
# filename = home_path + "/examples/hatefulmemes/unimodal_img.py"
# exec(compile(open(filename, "rb").read(), filename, 'exec'))

# print("------------- UNIMODAL TXT -------------")
# filename = home_path + "/examples/hatefulmemes/unimodal_txt.py"
# exec(compile(open(filename, "rb").read(), filename, 'exec'))

# print("------------- LATE FUSION -------------")
# filename = home_path + "/examples/hatefulmemes/late_fusion.py"
# exec(compile(open(filename, "rb").read(), filename, 'exec'))

# print("------------- LOW RANK TENSOR FUSION -------------")
# filename = home_path + "/examples/hatefulmemes/low_rank_tensor_fusion.py"
# exec(compile(open(filename, "rb").read(), filename, 'exec'))

# print("------------- MULTI INTERAC MATRIX -------------")
# filename = home_path + "/examples/hatefulmemes/multi_interac_matrix.py"
# exec(compile(open(filename, "rb").read(), filename, 'exec'))

# print("------------- TENSOR FUSION -------------")
# filename = home_path + "/examples/hatefulmemes/tensor_fusion.py"
# exec(compile(open(filename, "rb").read(), filename, 'exec'))


def objective(trial):
    # Define the hyperparameters to optimize
    configuration['Hyperparameters']['epochs'] = trial.suggest_int("epochs",10, 10, log =True)
    # configuration["Dataset"]["batch_size"] = trial.suggest_int("batch_size",64, 64, log =True)
    configuration["Dataset"]["batch_size"] = trial.suggest_categorical("batch_size", [32,64])
    configuration["Models"]["regularization"] = trial.suggest_categorical("regularization", [False, True])
    num_hidden = trial.suggest_int("num_hidden",1, 5, log =True)
    configuration["Models"]["mlp_hidden_sizes"] = [trial.suggest_categorical(f"mlp_hidden_sizes_{i}", [32,64,128,256]) for i in range(num_hidden)]
    configuration["Models"]["mlp_dropout_prob"] = [trial.suggest_float(f"mlp_dropout_prob_{i}", 0.1, 0.9, log = True) for i in range(num_hidden)]
    configuration["Optimizers"]["optimizer"] = trial.suggest_categorical("optimizer", ['Adam'])
    configuration['Optimizers']['learning_rate'] = trial.suggest_float("learning_rate", 1e-2, 1e-1, log=True)
    configuration['Optimizers']['use_lr_scheduler'] = trial.suggest_categorical("use_lr_scheduler", [False])
    configuration['Optimizers']['momentum'] = trial.suggest_float("momentum", 0.1, 0.99, log=True)
    configuration["Loss"]["loss_fn"] = trial.suggest_categorical("loss_fn", ['BCELoss'])
    configuration["Loss"]["reg_lambda"] = trial.suggest_float("reg_lambda", 0.00001, 0.1, log=True)

    print("\n\nHyperparameters: ", trial.params)

    print("------------- LATE FUSION -------------")
    train_metrics = late_fusion(configuration,trial)

    # print("------------- LOW RANK TENSOR FUSION -------------")
    # train_metrics = low_rank_tensor_fusion(configuration,trial)

    # print("------------- UNIMODAL IMG -------------")
    # train_metrics = unimodal_img(configuration,trial)

    # print("------------- UNIMODAL TXT -------------")
    # train_metrics = unimodal_txt(configuration,trial)

    train_metrics = pd.DataFrame(train_metrics)
    val_acc = train_metrics.loc[train_metrics.shape[0]-1,"val_acc"]
    val_loss = train_metrics.loc[train_metrics.shape[0]-1,"val_loss"]
    val_f1 = train_metrics.loc[train_metrics.shape[0]-1,"val_f1score"]

    # results_path = '/home/nippani.a/Silvio/Multimodal/results/hp_tuning/early_fusion/'
    # results_path = '/home/nippani.a/Silvio/Multimodal/results/hp_tuning/early_low_rank_fusion/'
    # results_path = '/home/nippani.a/Silvio/Multimodal/results/hp_tuning/early_unimodal_img/'
    # results_path = '/home/nippani.a/Silvio/Multimodal/results/hp_tuning/early_unimodal_txt/'
    # results_path = '/home/nippani.a/Silvio/Multimodal/results/hp_tuning/late_unimodal_img/'
    # results_path = '/home/nippani.a/Silvio/Multimodal/results/hp_tuning/late_unimodal_txt/'
    results_path = '/home/nippani.a/Silvio/Multimodal/results/hp_tuning/late_fusion/'
    # results_path = '/home/nippani.a/Silvio/Multimodal/results/hp_tuning/late_low_rank_fusion/'


    train_metrics.to_csv(results_path + "metrics/history_" + str(trial.number) + ".csv")

    plot_loss_accuracy(train_metrics, trial.params, results_path + "plots/history_" + str(trial.number) + ".png")

    return val_acc, val_loss, val_f1

study = optuna.create_study(directions=["maximize","minimize", "maximize"], sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
start_time = time.time()
study.optimize(objective, n_trials=3)
end_time = time.time()

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print(" Number of finished trials: ", len(study.trials))
print(" Number of pruned trials: ", len(pruned_trials))
print(" Number of complete trials: ", len(complete_trials))
# print("Best trial:")
# trial = study.best_trials
# print(" Value: ", trial.value)

# # Print the best hyperparameters and validation accuracy
# print(f"Best trial: {study.best_trial.number}")
# print(f"Best validation f1 score: {study.best_trial.value}")
# print(f"Best hyperparameters: {study.best_trial.params}")

def trial_params(trial):
    params = {}
    for key, value in trial.params.items():
        if isinstance(value, (int, float, str)):
            params[key] = [value]
    return pd.DataFrame(params)

# Collect the parameters for each trial in the study
params_list = []
for trial in study.trials:
    params_list.append(trial_params(trial))

# Concatenate the parameters into a single DataFrame
params_df = pd.concat(params_list, axis=0)

# Print the resulting DataFrame
print(params_df)
# results_path = '/home/nippani.a/Silvio/Multimodal/results/hp_tuning/early_fusion/'
# results_path = '/home/nippani.a/Silvio/Multimodal/results/hp_tuning/early_low_rank_fusion/'
# results_path = '/home/nippani.a/Silvio/Multimodal/results/hp_tuning/early_unimodal_img/'
# results_path = '/home/nippani.a/Silvio/Multimodal/results/hp_tuning/early_unimodal_txt/'
# results_path = '/home/nippani.a/Silvio/Multimodal/results/hp_tuning/late_unimodal_img/'
# results_path = '/home/nippani.a/Silvio/Multimodal/results/hp_tuning/late_unimodal_txt/'
results_path = '/home/nippani.a/Silvio/Multimodal/results/hp_tuning/late_fusion/'
# results_path = '/home/nippani.a/Silvio/Multimodal/results/hp_tuning/late_low_rank_fusion/'


params_df.to_csv(results_path + "study_params.csv")


