import asyncio
import os
from fairchem.demo.ocpapi import find_adsorbate_binding_sites, keep_all_slabs, keep_slabs_with_miller_indices
from utils import * 

system_id =  "16_7283_37" #"71_2537_62"
metadata_path = "/home/hoon/llm-agent/adsorb/data/processed/updated_sid_to_details.pkl"
save_path = "results-ocpdemo"
info = load_info_from_metadata(system_id, metadata_path)
mpid, miller, shift, top, ads, cat = info
# breakpoint()

async def main():
    results = await find_adsorbate_binding_sites(adsorbate=ads, 
                                                 bulk=mpid, 
                                                 adslab_filter=keep_slabs_with_miller_indices([miller]))
    
    # results = await find_adsorbate_binding_sites(adsorbate='*NH2', 
    #                                              bulk='mp-1103212', 
    #                                              adslab_filter=keep_slabs_with_miller_indices([(1, 2, 1)]))
    
    with open(os.path.join(save_path, f"{system_id}.json"), 'w') as f:
        f.write(results.to_json())

# Run the async function
asyncio.run(main())

