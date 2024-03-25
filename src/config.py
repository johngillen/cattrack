import yaml 

def read_yaml_file(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)

config = read_yaml_file('config.yaml')
