import asyncio
from fairchem.demo.ocpapi import find_adsorbate_binding_sites, keep_all_slabs, keep_slabs_with_miller_indices


async def main():
    results = await find_adsorbate_binding_sites(adsorbate='*OH', bulk='mp-126', adslab_filter=keep_slabs_with_miller_indices([(1,0,0)]))
    
    with open('results.json', 'w') as f:
        f.write(results.to_json())

# Run the async function
asyncio.run(main())