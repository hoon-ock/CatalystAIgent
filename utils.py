import pandas as pd
import os, yaml, pickle, shutil, json, ast
from fairchem.core.models.model_registry import model_name_to_local_file
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from ase.optimize import BFGS


def load_config(config_file):
    """Load configuration settings from a YAML file and print them."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    # Print the loaded configuration
    print("Loaded Configuration:")
    print(config_file)
    print(yaml.dump(config, default_flow_style=False))
    
    return config

def setup_save_path(config, duplicate=True):
    """Set up and return a unique save path based on the configuration name."""
    
    # Extract base save directory and configuration name (without file extension)
    # config_name = os.path.splitext(config['config_name'])[0]
    save_dir = config['paths']['save_dir']
    
    # Construct the initial save path
    save_path = os.path.join(save_dir, config['config_name'])
    
    if duplicate:
        # Ensure the save path is unique by appending an index if necessary
        if os.path.exists(save_path):
            i = 1
            while os.path.exists(f"{save_path}_{i}"):
                i += 1
            save_path = f"{save_path}_{i}"

        # Create the directory if it does not exist

    os.makedirs(save_path, exist_ok=True)
    
    return save_path

# def setup_paths(system_info, paths, mode='agent'):
#     """
#     system_info: system information in config yaml
#     paths: paths in config yaml
#     mode: 'agent' or 'ocp'
#     """
#     system_id = system_info.get('system_id', None)
#     if system_id is None:
#         ads = system_info['ads_smiles']
#         bulk_id = system_info['bulk_id']
#         bulk_symbol = system_info['bulk_symbol']
#         miller = str(system_info['miller']).replace(" ", "")
#         shift = str(system_info.get('shift', 'NA'))
#         #system_id = f"{system_info['ads_smiles']}_{system_info['bulk_symbol']}_{system_info['bulk_id']}_{str(system_info['miller'])}_{str(system_info['shift'])}"
        
#     else:
#         metadata_path = paths['metadata_path']
#         info = load_info_from_metadata(system_id, metadata_path)
#         bulk_id, miller, shift, top, ads, bulk_symbol = info
#         #bulk_id, miller, _, _, ads, bulk_symbol = info
#         if mode == 'ocp':
#             shift = 'NA'
#         miller = str(miller).replace(" ", "")
#     save_name = f"{ads}_{bulk_symbol}_{bulk_id}_{miller}_{shift}"
#     save_dir = paths['save_dir']
#     # tag = "llm" if mode == "llm-guided" else "llm_heur"
#     save_path = os.path.join(save_dir, f"{save_name}")
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     else:
#         # make copy version
#         i = 1
#         while os.path.exists(f"{save_path}_{i}"):
#             i += 1
#         save_path = f"{save_path}_{i}"
    
#     return save_path
    #return os.path.join(save_dir, f"{system_id}")


def load_system_info(system_info, metadata_path):
    system_id = system_info.get('system_id', None)
    if system_id is None:
        ads = system_info['ads_smiles']
        bulk_id = system_info['bulk_id']
        bulk_symbol = system_info['bulk_symbol']
        miller = str(system_info['miller'])
        shift = system_info['shift']
        top = system_info['top']
    else:
        # metadata_path = paths['metadata_path']
        bulk_id, miller, shift, top, ads, bulk_symbol = load_info_from_metadata(system_id, metadata_path)
    if not isinstance(miller, tuple):
        miller = ast.literal_eval(miller)
    print(f"bulk_id: {bulk_id}, miller: {miller}, shift: {shift}, top: {top}, ads: {ads}, bulk_symbol: {bulk_symbol}")
    return bulk_id, miller, shift, top, ads, bulk_symbol
 

def load_metadata(metadata_path, system_id):
    """Load metadata or extract system ID list if applicable."""
    if system_id == "all":
        metadata = pd.read_pickle(metadata_path)
        return list(metadata.keys()), metadata
    return [system_id], None

def save_result(result, config, save_dir):
    """Save the result as a pickle file and copy configuration for record."""
    result_path = os.path.join(save_dir, 'result.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump(result, f)

    # Save it as text file
    result_path = os.path.join(save_dir, 'result.txt')
    with open(result_path, 'w') as f:
        f.write(str(result))
    # result_path = os.path.join(save_dir, 'result.json')
    # # Save result in JSON format
    # with open(result_path, 'w') as f:
    #     json.dump(result, f)  # indent=4 for pretty printing
    
    
    # save config in the directory
    config_name = config['config_name']
    config_path = os.path.join(save_dir, f'{config_name}.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)


    # Copy config file for reproducibility
    # shutil.copy('config/adsorb_aigent.yaml', save_dir)



def derive_input_prompt(system_info, metadata_path):
    system_id = system_info.get("system_id", None)
    # breakpoint()
    if system_id is not None:
        sid_to_details = pd.read_pickle(metadata_path)
        miller = sid_to_details[system_id][1]
        ads = sid_to_details[system_id][4] #.replace("*", "")  
        cat = sid_to_details[system_id][5]
        
    else:
        miller = system_info.get("miller", None)
        ads = system_info.get("ads_smiles", None)
        cat = system_info.get("bulk_symbol", None)

    assert ads is not None and cat is not None and miller is not None, "Missing system information."
    
    prompt = f"The adsorbate is {ads} and the catalyst surface is {cat} {miller}."
    return prompt

def load_text_file(file_path):
    """Loads a text file and returns its content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"An error occurred: {e}"

def load_info_from_metadata(system_id, metadata_path):
    '''
    metadata: sid_to_details dictionary

    need to update the function to return AdsorbateSlabConfig object
    '''
    # breakpoint()
    metadata = pd.read_pickle(metadata_path)
    bulk_id = metadata[system_id][0]
    miller = metadata[system_id][1]
    shift = metadata[system_id][2]
    top = metadata[system_id][3]
    ads = metadata[system_id][4] #.replace("*", "")  
    bulk = metadata[system_id][5]
    return bulk_id, miller, shift, top, ads, bulk

def relax_adslab(adslab, model_name, save_path):
    checkpoint_path = model_name_to_local_file(model_name, local_cache='/tmp/fairchem_checkpoints/')
    calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=False)
    adslab.calc = calc
    opt = BFGS(adslab, trajectory=save_path)
    opt.run(fmax=0.05, steps=100)    
    return adslab