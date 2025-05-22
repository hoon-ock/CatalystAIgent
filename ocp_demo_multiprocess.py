import os
import yaml
import glob
import numpy as np
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from fairchem.demo.ocpapi import find_adsorbate_binding_sites, keep_slabs_with_miller_indices
import asyncio
from utils import * 

config = load_config('config/adsorb_agent.yaml')
paths = config['paths']
system_path = paths['system_dir']
system_config_files = glob.glob(system_path + '/*.yaml')
system_config_files.sort()

def process_config(config_file):
    config_name = os.path.basename(config_file).split('.')[0]
    org_config = load_config(config_file)
    config = org_config.copy()
    config['paths'] = paths
    config['config_name'] = config_name
    save_dir = setup_save_path(config, duplicate=False)
    # if save_dir already exists, skip this config
    if os.path.exists(save_dir):
        print(f"Skip: {config_name} already exists")
        return

    system_info = config['system_info']
    metadata_path = paths['metadata_path']
    try:
        bulk_id, miller, shift, top, ads, bulk_symbol = load_system_info(system_info, metadata_path)
    except Exception as e:
        print(f"Error: {config_name} is not a valid config file – {e}")
        return

    try:
        # Call sync wrapper for async function
        results = run_find_adsorbate_binding_sites_sync(adsorbate=ads, bulk=bulk_id, miller=miller)

        slab_selected = None
        for slab in results.slabs:
            if (np.isclose(slab.slab.metadata.shift, shift, atol=0.01) and 
                slab.slab.metadata.top == top):
                slab_selected = slab
                org_config['system_info']['num_site'] = len(slab_selected.configs)
                config['system_info']['num_site'] = len(slab_selected.configs)
                break

        if slab_selected is None:
            print(f"Warning: No matching slab found for {config_name}")
            return

        with open(os.path.join(save_dir, "full_result.json"), 'w') as f:
            f.write(results.to_json())

        results.slabs = [slab_selected]
        with open(os.path.join(save_dir, "selected_result.json"), 'w') as f:
            f.write(results.to_json())

        with open(config_file, 'w') as f:
            yaml.dump(org_config, f)

        with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)

    except Exception as e:
        print(f"Failed to process {config_name}: {e}")


# Wrap async function to make it blocking for multiprocessing
def run_find_adsorbate_binding_sites_sync(adsorbate, bulk, miller):
    

    async def inner():
        return await find_adsorbate_binding_sites(
            adsorbate=adsorbate,
            bulk=bulk,
            adslab_filter=keep_slabs_with_miller_indices([miller])
        )

    return asyncio.run(inner())



max_workers = 4 #os.cpu_count() or 4

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(process_config, cfg) for cfg in system_config_files]

    for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
        try:
            future.result()
        except Exception as e:
            print(f"Error in task {i}: {e}")

        # Optional pause every 10
        if (i + 1) % 10 == 0:
            print("Pausing...")
            time.sleep(10)