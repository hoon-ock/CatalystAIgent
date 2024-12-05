# import sys
# sys.path.append('../')

from ase.visualize.plot import plot_atoms
import matplotlib.pyplot as plt
import os
import glob
import tqdm
import pandas as pd
import numpy as np
import ase.io
from fairchem.demo.ocpapi import AdsorbateBindingSites


def system_finder(config_path):
    config_list = glob.glob(f"{config_path}/*.yaml")
    config_name_list = [os.path.basename(config).split('.')[0] for config in config_list]
    config_name_list.sort()
    return config_name_list

def result_path_finder(result_mother_path, config_name_list):
    path_list = glob.glob(os.path.join(result_mother_path+'_SET*'))   
    # Initialize the result path list
    result_path_list = []
    # Loop through each directory and append config paths
    for path in path_list:
        for config_name in config_name_list:
            result_path_list.append(os.path.join(path, config_name))
    result_path_list.sort()
    return result_path_list

def get_results(result_path, mode='agent'):
    """
    Fetches results based on the specified mode.

    Parameters:
    - result_path (str): The path to the directory containing the results.
    - mode (str): The mode to fetch results. Options are 'agent' or 'ocpdemo'. Default is 'agent'.

    Returns:
    - tuple: A tuple containing result data, trajectories (if applicable), and valid energies.
    """
    try:
        if mode == 'agent':
            # Load result.pkl
            result = pd.read_pickle(os.path.join(result_path, 'result.pkl'))
            
            # Load trajectories
            trajs = glob.glob(os.path.join(result_path, 'traj', '*.traj'))
            
            # Load valid energies
            valid_energies = pd.read_pickle(os.path.join(result_path, 'valid_energies.pkl'))
            
            return result, trajs, valid_energies

        elif mode == 'ocpdemo':
            # Load selected_result.json
            selected_result_path = os.path.join(result_path, 'selected_result.json')
            with open(selected_result_path, 'r') as f:
                result = AdsorbateBindingSites.from_json(f.read())
            
            # Load valid energies
            valid_energies = pd.read_pickle(os.path.join(result_path, 'valid_energies.pkl'))
            
            return result, None, valid_energies

        else:
            raise ValueError(f"Invalid mode: {mode}. Supported modes are 'agent' and 'ocpdemo'.")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None, None, None
    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")
        return None, None, None
    
def get_min_energy(valid_energies):
    min_energy = 1e10
    for idx, energy in valid_energies.items():
        if energy < min_energy:
            min_energy = energy
            if '_' in str(idx):
                min_idx = int(idx.split('_')[1])
            else:
                min_idx = int(idx)
    return min_energy, min_idx

def get_matching_traj(result_path, min_energy, min_idx):
    #trajs = glob.glob(os.path.join(result_path, 'traj', '*.traj'))
    min_traj_path = os.path.join(result_path, f'traj/config_{min_idx}.traj')
    min_traj = ase.io.read(min_traj_path, ':')
    initial_image = min_traj[0]
    relaxed_image = min_traj[-1]
    relaxed_energy = relaxed_image.get_potential_energy()
    assert np.isclose(relaxed_energy, min_energy, 0.01), f"Relaxed energy ({relaxed_energy}) does not match minimum energy ({min_energy})."
    return initial_image, relaxed_image


def get_ocpdemo_structure(result, min_idx):
    relaxed_image = result.slabs[0].configs[min_idx].to_ase_atoms()
    return relaxed_image


def adslab_plot(initial_image, relaxed_image, save_dir, mode):
    if initial_image is not None and relaxed_image is not None:
        # visualize initial and final structures
        fig, ax = plt.subplots(1, 2)
        labels = ['Initial', 'Relaxed']
        for i in range(2):
            ax[i].axis('off')
            ax[i].set_title(labels[i])
        plot_atoms(initial_image, ax[0], radii=0.8, rotation=("-75x, 45y, 10z"))
        plot_atoms(relaxed_image, ax[1], radii=0.8, rotation=("-75x, 45y, 10z"))

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{mode}_adslabs.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    if relaxed_image is not None:
        fig, ax = plt.subplots()
        ax.axis('off')
        plot_atoms(relaxed_image, ax, radii=0.8, rotation=("-75x, 45y, 10z"))
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{mode}_relaxed_adslab.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='Path to the directory containing the config files.')
    parser.add_argument('--result_mother_path', type=str, required=True, help='Path to the directory containing the results.')
    parser.add_argument('--mode', choices=['agent', 'ocpdemo'], default='agent', help='Mode to fetch results. Options are agent or ocpdemo.')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to the directory to save the visualizations.')
    args = parser.parse_args()

    config_path = args.config_path  
    result_mother_path = args.result_mother_path
    mode = args.mode
    save_dir = args.save_dir
    # create warning message if mode is not included in result_mother_path
    if mode not in result_mother_path:
        print(f"Warning: Mode '{mode}' not found in result_mother_path '{result_mother_path}'.")

    config_name_list = system_finder(config_path)
    result_path_list = result_path_finder(result_mother_path, config_name_list)
    # breakpoint()
    for result_path in tqdm.tqdm(result_path_list):
        set_no, system = result_path.split('_SET')[-1].split('/')
        save_path = os.path.join(save_dir, system, f'SET{set_no}')
        os.makedirs(save_path, exist_ok=True)
        result, trajs, valid_energies = get_results(result_path, mode=mode)
        min_energy, min_idx = get_min_energy(valid_energies)
        if mode == 'agent':
            initial_image, relaxed_image = get_matching_traj(result_path, min_energy, min_idx)
        elif mode == 'ocpdemo':
            relaxed_image = get_ocpdemo_structure(result, min_idx)
            initial_image = None
        adslab_plot(initial_image, relaxed_image, save_path, mode)



