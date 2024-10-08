import yaml
import importlib
import os
import torch

home_path = os.getcwd()

# Load the YAML file
with open(home_path + '/config/inference/late_fusion.yaml') as f:
    configuration = yaml.safe_load(f)

configuration['General']['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_import_package = importlib.import_module(configuration['Models']['base_import_package'])

configuration['Dataset']['best_model_path'] = home_path + configuration['Dataset']['best_model_path']
configuration['Dataset']['train_metrics_path'] = home_path + configuration['Dataset']['train_metrics_path']
configuration['Dataset']['test_metrics_path'] = home_path + configuration['Dataset']['test_metrics_path']

configuration['Models']['image_processor'] = getattr(base_import_package,configuration['Models']['image_processor_package'])
configuration['Models']['image_processor'] = configuration['Models']['image_processor'].from_pretrained(configuration['Models']['image_processor_pretrained'])

configuration['Models']['image_model'] = getattr(base_import_package,configuration['Models']['image_model_package'])
configuration['Models']['image_model'] = configuration['Models']['image_model'].from_pretrained(configuration['Models']['image_model_name'])

configuration['Models']['text_tokenizer'] = getattr(base_import_package,configuration['Models']['text_tokenizer_package'])
configuration['Models']['text_tokenizer'] = configuration['Models']['text_tokenizer'].from_pretrained(configuration['Models']['text_processor_pretrained'])

configuration['Models']['text_model'] = getattr(base_import_package,configuration['Models']['text_model_package'])
configuration['Models']['text_model'] = configuration['Models']['text_model'].from_pretrained(configuration['Models']['text_model_name'])


filename = home_path + "/examples/inference/inference_dataset_pipeline.py"
exec(compile(open(filename, "rb").read(), filename, 'exec'))





