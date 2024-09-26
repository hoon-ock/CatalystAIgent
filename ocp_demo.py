import asyncio
import os
import ast
from fairchem.demo.ocpapi import find_adsorbate_binding_sites, keep_all_slabs, keep_slabs_with_miller_indices
from utils import * 


config = load_config('config/ocp_demo.yaml')
system_info = config['system_info']
system_id = system_info.get('system_id', None)

paths = config['paths']
save_dir = paths['save_dir']
if system_id is None:
    ads = system_info['ads_smiles']
    bulk_id = system_info['bulk_id']
    bulk_symbol = system_info['bulk_symbol']
    miller = system_info['miller']
    shift = system_info['shift']
    if not isinstance(miller, tuple):
        miller = ast.literal_eval(miller)
    file_name = f"{ads}_{bulk_symbol}_{bulk_id}_{str(miller)}_{str(shift)}.json"
else:
    metadata_path = paths['metadata_path']
    info = load_info_from_metadata(system_id, metadata_path)
    bulk_id, miller, shift, top, ads, bulk_symbol = info
    file_name = f"{system_id}.json"


save_path = os.path.join(save_dir, file_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


async def main():
    results = await find_adsorbate_binding_sites(adsorbate=ads, 
                                                 bulk=bulk_id, 
                                                 adslab_filter=keep_slabs_with_miller_indices([miller]))
    
    with open(save_path, 'w') as f:
        f.write(results.to_json())

# Run the async function
asyncio.run(main())

