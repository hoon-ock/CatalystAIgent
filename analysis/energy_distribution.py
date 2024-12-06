import matplotlib.pyplot as plt
import os
import tqdm
import pandas as pd
from fairchem.demo.ocpapi import AdsorbateBindingSites
from visualizer import system_finder

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True, help='Path to the directory containing the config files.')
parser.add_argument('--save_dir', type=str, required=True, help='Path to the directory to save the plot.')
args = parser.parse_args()
config_path = args.config_path
save_dir = args.save_dir

config_name_list = system_finder(config_path)

for config_name in tqdm.tqdm(config_name_list):


    agent_valid_energy_list = []
    ocpdemo_valid_energy_list = []
    for i in range(3):
        llm_path = f"/home/hoon/llm-agent/results/agent/analysis_SET{i+1}/{config_name}" 
        energies = pd.read_pickle(os.path.join(llm_path, 'valid_energies.pkl'))
        agent_valid_energy_list += list(energies.values())

        ocpdemo_path = f"/home/hoon/llm-agent/results/ocpdemo/analysis_SET{i+1}/{config_name}"
        with open(os.path.join(ocpdemo_path, 'selected_result.json'), 'r') as f:
            demo_results = AdsorbateBindingSites.from_json(f.read())
        demo_valid_energies = pd.read_pickle(os.path.join(ocpdemo_path, 'valid_energies.pkl'))
        ocpdemo_valid_energy_list+= list(demo_valid_energies.values())


    # plot the distribution of the energies
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the valid energy distribution with distinct colors
    ax.hist(list(ocpdemo_valid_energy_list), bins=20, alpha=0.7, color='salmon', label='Algorithms', edgecolor='black')
    ax.hist(list(agent_valid_energy_list), bins=20, alpha=0.7, color='steelblue', label='Adsorb-Agent', edgecolor='black')

    # Set axis labels
    ax.set_xlabel('Energy [eV]', fontsize=20, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=20, fontweight='bold')

    # Set tick parameters for better visibility
    ax.tick_params(axis='both', which='major', labelsize=18)

    # Display the legend
    ax.legend(fontsize=20)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the figure with high dpi for clarity
    save_path = os.path.join(save_dir, config_name, 'energy_dist.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)