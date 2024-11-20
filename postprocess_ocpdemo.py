from fairchem.demo.ocpapi import AdsorbateBindingSites
from fairchem.data.oc.core import Adsorbate, Bulk, Slab, AdsorbateSlabConfig
from tools import DetectTrajAnomaly
import numpy as np
import pickle, os, glob, ast
from tqdm import tqdm
from utils import * 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, help='path to the results directory')
args = parser.parse_args()
path = args.dir

# path = "/home/hoon/llm-agent/results/ocpdemo"

result_dirs = os.listdir(path)
result_dirs.sort()
for result_dir in result_dirs:
    result_path = os.path.join(path, result_dir, 'selected_result.json')
    save_path = os.path.join(path, result_dir)
    config_file_path = os.path.join(path, result_dir, 'config.yaml')

    config_file = load_config(config_file_path)
    paths = config_file['paths']
    ads_db_path = paths['ads_db_path']
    bulk_db_path = paths['bulk_db_path']
    system_info = config_file['system_info']
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


    
    # full_path = os.path.join(path, result_dir, 'full_result_mod.json')
    # with open(full_path, 'r') as f:
    #     full_results = AdsorbateBindingSites.from_json(f.read())

    with open(result_path, 'r') as f:
        results = AdsorbateBindingSites.from_json(f.read())
    
    status = {}
    valid_energy = {}
    slab_selected = results.slabs[0]
    print(slab_selected.slab.metadata)
    assert slab_selected.slab.metadata.bulk_src_id == bulk_id, f"Bulk mismatch: {slab_selected.slab.metadata.bulk_src_id} != {bulk_id}" 
    assert slab_selected.slab.metadata.top == top, f"Top mismatch: {slab_selected.slab.metadata.top} != {top}"
    assert np.isclose(slab_selected.slab.metadata.shift, shift, atol=0.01), f"Shift mismatch: {slab_selected.slab.metadata.shift} != {shift}"

    # set up adsorbate and bulk to create the initial structures for anomaly detection
    adsorbate = Adsorbate(adsorbate_smiles_from_db=ads, adsorbate_db_path=ads_db_path)
    bulk = Bulk(bulk_src_id_from_db=bulk_id, bulk_db_path=bulk_db_path)
    slabs = Slab.from_bulk_get_specific_millers(bulk=bulk, specific_millers=miller)
    for slab_candidate in slabs:
        if np.isclose(slab_candidate.shift, shift, atol=0.01) and slab_candidate.top == top:
            # slab_selected = slab_candidate
            break

    adslabs_ = AdsorbateSlabConfig(slab_candidate, adsorbate, mode='heuristic') #num_sites=1,
    adslabs = [*adslabs_.atoms_list]
    init_image = adslabs[0]
    
    config_idx = 0
    for config in tqdm(slab_selected.configs, desc=f"Processing {result_dir}"):
        fin_image = config.to_ase_atoms()
        tags = fin_image.get_tags()
        anomaly_detector = DetectTrajAnomaly(init_image, fin_image, tags, surface_change_cutoff_multiplier=2.0)
        dissoc = anomaly_detector.is_adsorbate_dissociated()
        desorb = anomaly_detector.is_adsorbate_desorbed()
        recon = anomaly_detector.has_surface_changed()
        if not dissoc and not desorb and not recon:
            valid_energy[config_idx] = config.energy
            status[config_idx] = 'normal'
        else:
            if dissoc:
                status[config_idx] = 'dissociated'
            if desorb:
                status[config_idx] = 'desorbed'
            if recon:
                status[config_idx] = 'reconstructed'
        config_idx += 1

    # Minimum energy
    if len(valid_energy) == 0:
        print(f'No valid energies found: {result_dir}')
        continue
    min_energy = min(valid_energy.values())
    min_config = [key for key, value in valid_energy.items() if value == min_energy][0]
    print(f'Minimum energy: {min_energy}')
    print(f'Config index: {min_config}')
    # breakpoint()

    # Save it as a text file
    with open(os.path.join(save_path, 'min_energy.txt'), 'w') as f:
        f.write(f'Minimum energy: {min_energy}\n')
        f.write(f'Config index: {min_config}\n')
        f.write(f'Metadata: {slab_selected.slab.metadata}')
    # Save valid energies
    with open(os.path.join(save_path, 'valid_energies.pkl'), 'wb') as f:
        pickle.dump(valid_energy, f)