from tools import DetectTrajAnomaly
from ase.io import read
import glob
import os
import pickle
from tqdm import tqdm
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument("--dir", type=str, default="results")
args = argparser.parse_args()


# List all directories in the results folder
dir_list = glob.glob(f"{args.dir}/*")
# breakpoint()
assert len(dir_list) > 0, f"No directories found in {args.dir}"

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

    print(f"Number of traj files: {len(traj_files)}")
    print(f"Number of valid energies: {len(valid_energies)}")

    # Save valid energies if any are found
    if valid_energies:
        output_path = os.path.join(dir, "valid_energies.pkl")
        min_energy = min(valid_energies.values())
        min_config = [key for key, value in valid_energies.items() if value == min_energy]

        with open(output_path, 'wb') as pickle_file:
            pickle.dump(valid_energies, pickle_file)

        print(f"Minimum energy: {min_energy} for {min_config}")
        # save this as a text file
        with open(os.path.join(dir, "min_energy.txt"), "w") as f:
            f.write(f"Minimum energy: {min_energy} for {min_config}")

        #print(f"Valid energies saved to {output_path}")
    else:
        print(f"No valid energies found for {dir}")