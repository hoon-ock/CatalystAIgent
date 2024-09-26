import pandas as pd
import os, yaml, pickle, shutil, json
from fairchem.core.models.model_registry import model_name_to_local_file
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from ase.optimize import BFGS

def load_config(config_file):
    """Load configuration settings from a YAML file and print them."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    # Print the loaded configuration
    print("Loaded Configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    return config

def setup_paths(system_info, mode, paths):
    """Prepare save directory based on system ID and mode."""
    system_id = system_info.get('system_id', None)
    if system_id is None:
        system_id = f"{system_info['ads_smiles']}_{system_info['bulk_symbol']}_{system_info['bulk_id']}_{str(system_info['miller'])}_{str(system_info['shift'])}"
    
    save_dir = paths['save_dir']
    tag = "llm" if mode == "llm-guided" else "llm_heuristic"
    return os.path.join(save_dir, f"{system_id}_{tag}")

def load_metadata(metadata_path, system_id):
    """Load metadata or extract system ID list if applicable."""
    if system_id == "all":
        metadata = pd.read_pickle(metadata_path)
        return list(metadata.keys()), metadata
    return [system_id], None

def save_result(result, save_dir):
    """Save the result as a pickle file and copy configuration for record."""
    # result_path = os.path.join(save_dir, 'result.pkl')
    # with open(result_path, 'wb') as f:
    #     pickle.dump(result, f)

    result_path = os.path.join(save_dir, 'result.json')
    # Save result in JSON format
    with open(result_path, 'w') as f:
        json.dump(result, f)  # indent=4 for pretty printing
    
    # Copy config file for reproducibility
    shutil.copy('config.yaml', save_dir)



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