import asyncio
import os
import ast
import glob
import numpy as np
from fairchem.data.oc.core import Slab
from fairchem.demo.ocpapi import find_adsorbate_binding_sites, keep_slabs_with_miller_indices
from utils import * 


config = load_config('config/adsorb_agent.yaml')
paths = config['paths']
system_path = paths['system_dir']
system_config_files = glob.glob(system_path + '/*.yaml')

for config_file in system_config_files:
    config_name = os.path.basename(config_file)
    config_name = config_name.split('.')[0]
    org_config = load_config(config_file)
    config = org_config.copy()
    config['paths'] = paths
    config['config_name'] = config_name
    save_dir = setup_save_path(config)


    system_info = config['system_info']
    system_id = system_info.get('system_id', None)
    if system_id is None:
        ads = system_info['ads_smiles']
        bulk_id = system_info['bulk_id']
        bulk_symbol = system_info['bulk_symbol']
        miller = str(system_info['miller'])
        shift = system_info['shift']
        top = system_info['top']
    else:
        metadata_path = paths['metadata_path']
        bulk_id, miller, shift, top, ads, bulk_symbol = load_info_from_metadata(system_id, metadata_path)
        
    if not isinstance(miller, tuple):
        miller = ast.literal_eval(miller)


    async def main():
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
        

        assert slab_selected is not None, f"Slab with miller {miller} and shift {shift} not found"
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

