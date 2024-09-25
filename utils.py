import pandas as pd
import os, yaml, pickle, shutil

def load_config(config_file):
    """Load configuration settings from a YAML file and print them."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    # Print the loaded configuration
    print("Loaded Configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    return config

def setup_paths(system_id, mode, paths):
    """Prepare save directory based on system ID and mode."""
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
    result_path = os.path.join(save_dir, 'result.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump(result, f)
    
    # Copy config file for reproducibility
    shutil.copy('config.yaml', save_dir)



def derive_input_prompt(system_id, metadata_path):
    sid_to_details = pd.read_pickle(metadata_path)
    miller = sid_to_details[system_id][1]
    ads = sid_to_details[system_id][4].replace("*", "")  
    cat = sid_to_details[system_id][5]
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
    mpid = metadata[system_id][0]
    miller = metadata[system_id][1]
    shift = metadata[system_id][2]
    top = metadata[system_id][3]
    ads = metadata[system_id][4] #.replace("*", "")  
    cat = metadata[system_id][5]
    return mpid, miller, shift, top, ads, cat