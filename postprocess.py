from tools import DetectTrajAnomaly
from ase.io import read
import glob
import os
import pickle
from tqdm import tqdm

# List all directories in the results folder
dir_list = glob.glob("results/*")

for dir in dir_list:
    traj_dir = os.path.join(dir, "traj/")
    
    # Check if traj_dir exists before proceeding
    if not os.path.exists(traj_dir):
        print(f"Directory {traj_dir} does not exist, skipping.")
        continue
    
    # Collect all .traj files
    traj_files = glob.glob(os.path.join(traj_dir, "*.traj"))

    valid_energies = {}

    # Iterate over all traj files with a progress bar
    for traj_file in tqdm(traj_files, desc=f"Processing {os.path.basename(dir)}"):
        try:
            traj = read(traj_file, index=":")
            init_image = traj[0]
            fin_image = traj[-1]
            tags = init_image.get_tags()

            # Detect anomalies
            anomaly_detector = DetectTrajAnomaly(init_image, fin_image, tags)
            dissoc = anomaly_detector.is_adsorbate_dissociated()
            desorb = anomaly_detector.is_adsorbate_desorbed()
            recon = anomaly_detector.has_surface_changed()

            # If no anomalies, save the relaxed energy
            if not dissoc and not desorb and not recon:
                config_id = os.path.splitext(os.path.basename(traj_file))[0]
                relaxed_energy = fin_image.get_potential_energy()
                valid_energies[config_id] = relaxed_energy

        except Exception as e:
            print(f"Error processing {traj_file}: {e}")
            continue

    # Save valid energies if any are found
    if valid_energies:
        output_path = os.path.join(dir, "valid_energies.pkl")
        
        with open(output_path, 'wb') as pickle_file:
            pickle.dump(valid_energies, pickle_file)
        
        print(f"Valid energies saved to {output_path}")
    else:
        print(f"No valid energies found for {dir}")