import yaml
# from config_class import Config
import importlib
# from models.basic_models import MLP
import os

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
filename = home_path + "/examples/hatefulmemes/late_fusion.py"
exec(compile(open(filename, "rb").read(), filename, 'exec'))

# print("------------- LOW RANK TENSOR FUSION -------------")
# filename = home_path + "/examples/hatefulmemes/low_rank_tensor_fusion.py"
# exec(compile(open(filename, "rb").read(), filename, 'exec'))

# print("------------- MULTI INTERAC MATRIX -------------")
# filename = home_path + "/examples/hatefulmemes/multi_interac_matrix.py"
# exec(compile(open(filename, "rb").read(), filename, 'exec'))

# print("------------- TENSOR FUSION -------------")
# filename = home_path + "/examples/hatefulmemes/tensor_fusion.py"
# exec(compile(open(filename, "rb").read(), filename, 'exec'))






