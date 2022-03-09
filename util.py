import yaml
import os

def yaml_load():
    """ load config.yml

    Returns:
        [type]: [description]
    """
    with open("config.yml") as stream:
        param = yaml.safe_load(stream)
    return param

def verify_folders(path_input_folder, path_output_folder):
    """verify input and output folders

    Args:
        path_input_folder ([string]): [description]
        path_output_folder ([string]): [description]
    """
    
    # Check whether the specified path exists or not
    is_exist_input_folder = os.path.exists(path_input_folder) 
    is_exist_output_folder = os.path.exists(path_output_folder)
    
    if not is_exist_input_folder:
        # create a new directory because it does not exists
        os.makedirs(path_input_folder)
        print('[INFO]: Input folder created!')
        
    if not is_exist_output_folder:
        # create a new directory because it does not exists
        os.makedirs(path_output_folder)
        print('[INFO]: Output folder created!')
        

    
    