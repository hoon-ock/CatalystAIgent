from typing import List, Optional
import sys
from pathlib import Path
import numpy as np
import torch
import ast
import glob
from ase.io import read
from tools import SiteAnalyzer
from utils import *
import warnings
warnings.filterwarnings("ignore")
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def gnn_switch_relaxer(result_dir, gnn_model, new_result_dir):
    '''
    Run the relaxation on the LLM-selected initial configuraiotns with the designated GNN.
    This is used to relaxed the structures with the different GNNs from the one used in Adsorb-Agent.
    '''

    #########################################################################
    # if save_dir already exists, skip this config
    # check whether save_dir + /traj/*.traj exists
    system_id = result_dir.split('/')[-1]
    result_pkl_path = os.path.join(new_result_dir, system_id, 'result.pkl')
    if os.path.exists(result_pkl_path):
        print(f"Skip: {system_id} (result.pkl exists)")
        return None
    
    traj_dir = os.path.join(result_dir, "traj")
    # breakpoint()
    if os.path.exists(traj_dir):
        traj_files = glob.glob(os.path.join(traj_dir, '*.traj'))
        if len(traj_files) == 0:
            print(f"Skip: {system_id}")
            return None
    #########################################################################
    print("=" * 30)
    print(f"Relaxing system: {system_id}")
    print(f"Num of initial configurations: {len(glob.glob(os.path.join(traj_dir, '*.traj')))}")
    print("=" * 30)
    

    # if len(adslabs) == 0:
    #     print("No selected configurations even > 1.3 cutoff multiplier. Skipping to the next system.")

    #     return None


    
    ########### Geometry Optimizer ###########
    print("Relaxing adslabs...")

    # if not os.path.exists(traj_dir):
    #     os.makedirs(traj_dir)
    relaxed_energies = []
    for i, traj_file in enumerate(traj_files):
        adslab = read(traj_file, ':')[0] # init_image
    #for i, adslab in enumerate(adslabs):

        #save_path = os.path.join(traj_dir, f"config_{i}.traj")
        file_name = os.path.basename(traj_file)
        # new_save_path = os.path.join(traj_dir, file_name)
        new_traj_dir = os.path.join(new_result_dir, system_id, "traj")
        # breakpoint()
        os.makedirs(new_traj_dir, exist_ok=True)
        new_save_path = os.path.join(new_traj_dir, f"config_{i}.traj")
        with torch.no_grad():
            adslab = relax_adslab(adslab, gnn_model, new_save_path)
        relaxed_energies.append(adslab.get_potential_energy())

        
        #torch.cuda.empty_cache()  # clear cuda memory after each relaxation

    min_energy = np.min(relaxed_energies)
    min_idx = np.argmin(relaxed_energies)
    ###########################################
    # Open the existing result file in YAML format
    with open(os.path.join(result_dir, f'{system_id}.yaml'), "r") as f:
        config = yaml.safe_load(f)
    config['agent_settings']['gnn_model'] = gnn_model  
    # Read result.pkl (note: open in binary mode 'rb' for pickle files)
    with open(os.path.join(result_dir, 'result.pkl'), "rb") as f:
        result_dict = pickle.load(f)
    # breakpoint()
    # # Convert to dictionary
    # if result_dict['system'] != system_id:
    #     breakpoint()
    #     print(f"Error: system_id mismatch! {result_dict['system']} != {system_id}")
    #     return None

    result_dict['min_energy'] = min_energy
    result_dict['min_idx'] = min_idx


    print("Result:", result_dict)
    save_result(result_dict, config, os.path.join(new_result_dir, system_id))

    return result_dict


def multirun_switch_relaxer(prev_result_path, gnn_model, new_result_dir):

    system_id_list = os.listdir(prev_result_path)
    result_dirs = [os.path.join(prev_result_path, system_id) for system_id in system_id_list] 
    result_dirs.sort()
    for result_dir in result_dirs:
        try:
            gnn_switch_relaxer(result_dir, gnn_model, new_result_dir)
        except Exception as e:
            print(f"Error processing {result_dir}: {e}")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print('============ Completed! ============')




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Config file path')
    parser.add_argument('--prev_path', type=str)
    parser.add_argument('--gnn_model', type=str)
    parser.add_argument('--save_result_dir', type=str)
    args = parser.parse_args()

    
    multirun_switch_relaxer(args.prev_path, args.gnn_model, args.save_result_dir)

    