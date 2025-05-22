import asyncio
import os
import ast
import glob
import numpy as np
from fairchem.data.oc.core import Slab
from fairchem.demo.ocpapi import find_adsorbate_binding_sites, keep_slabs_with_miller_indices
from utils import * 
import time
init_time = time.time()
config = load_config('config/adsorb_agent.yaml')
paths = config['paths']
system_path = paths['system_dir']
system_config_files = glob.glob(system_path + '/*.yaml')
system_config_files.sort()

for i, config_file in enumerate(system_config_files):
    config_name = os.path.basename(config_file)
    config_name = config_name.split('.')[0]
    org_config = load_config(config_file)
    config = org_config.copy()
    config['paths'] = paths
    config['config_name'] = config_name
    save_dir = setup_save_path(config)
    # # if savd_dir already exists, skip this config
    # if os.path.exists(save_dir):
    #     breakpoint()
    #     print(f"Skip: {config_name} already exists")
    #     continue
    # else:
    #     os.makedirs(save_dir, exist_ok=True)

    # breakpoint()
    system_info = config['system_info']
    metadata_path = paths['metadata_path']
    try:
        bulk_id, miller, shift, top, ads, bulk_symbol = load_system_info(system_info, metadata_path)
    except:
        print(f"Error: {config_name} is not a valid config file")
        continue
    


    async def main():
        slab_selected = None
        results = await find_adsorbate_binding_sites(adsorbate=ads, 
                                                     bulk=bulk_id, 
                                                     adslab_filter=keep_slabs_with_miller_indices([miller]))
        for slab in results.slabs:
            slab_top = slab.slab.metadata.top
            slab_shift = slab.slab.metadata.shift
            if np.isclose(slab_shift, shift, atol=0.01) and slab_top == top:
                slab_selected = slab
                org_config['system_info']['num_site'] = len(slab_selected.configs)
                config['system_info']['num_site'] = len(slab_selected.configs)
                break
        

        # assert slab_selected is not None, f"Slab with miller {miller} and shift {shift} not found"
        # Check if slab_selected was found; if not, log and skip this config
        if slab_selected is None:
            print(f"Warning: No matching slab found for miller {miller} and shift {shift} in {config_name}")
            return  # Exit the `main` function for this config
        # breakpoint()
        with open(os.path.join(save_dir, "full_result.json"), 'w') as f:
            f.write(results.to_json())
        
        results.slabs = [slab_selected]
        with open(os.path.join(save_dir, "selected_result.json"), 'w') as f:
            f.write(results.to_json())
        
        # Update the original yaml with the number of adsorption configurations
        with open(config_file, 'w') as f:
            yaml.dump(org_config, f)
        
        # Save the config file for record
        with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)

    # Run the async function
    asyncio.run(main())

    # Pause every 10 iterations
    if (i + 1) % 10 == 0:
        print("Pausing for 5 seconds...")
        time.sleep(5)

fin_time = time.time()
print(f"Total time: {(fin_time - init_time)/60:.2f} minutes for {len(system_config_files)} systems")
print('============ Completed! ============')
