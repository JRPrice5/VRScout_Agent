#!/usr/bin/env python
"""
Batch VR Dataset Converter for ACT Training

This script extends convert_vr_dataset.py to process multiple VR datasets
organized in a directory structure. It expects a directory containing multiple
folders, where each folder has a CSV file and a subfolder with JPG images.
All datasets are combined as episodes of a single task in one output directory.

Usage:
    python batch_convert_vr_datasets.py --input_dir path/to/datasets --output_dir path/to/output --episode_length 100
"""

import os
import argparse
import glob
from pathlib import Path
from tqdm import tqdm
from convert_new_vr_dataset import convert_vr_dataset_to_hdf5, update_constants_file


def find_dataset_folders(input_dir):
    """
    Find all potential dataset folders in the input directory.
    A valid dataset folder must contain a CSV file in the /Poses subdirectory
    and JPG images in the /Pictures/leftEye subdirectory.
    
    Args:
        input_dir: Path to the parent directory containing dataset folders
        
    Returns:
        List of tuples (folder_path, csv_file, image_dir)
    """
    datasets = []
    
    # List all subdirectories in the input directory
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    for subdir in subdirs:
        folder_path = os.path.join(input_dir, subdir)
        
        # Check for CSV files in the /Poses subdirectory
        poses_dir = os.path.join(folder_path, "Poses")
        if not os.path.isdir(poses_dir):
            print(f"Warning: No Poses directory found in {folder_path}, skipping...")
            continue
            
        csv_files = glob.glob(os.path.join(poses_dir, "*.csv"))
        if not csv_files:
            print(f"Warning: No CSV file found in {poses_dir}, skipping...")
            continue
            
        # Use the first CSV file found
        csv_file = csv_files[0]
        
        # Check for JPG files in the /Pictures/leftEye subdirectory
        pictures_dir = os.path.join(folder_path, "Pictures")
        if not os.path.isdir(pictures_dir):
            print(f"Warning: No Pictures directory found in {folder_path}, skipping...")
            continue
            
        left_eye_dir = os.path.join(pictures_dir, "leftEye")
        if not os.path.isdir(left_eye_dir):
            print(f"Warning: No leftEye directory found in {pictures_dir}, skipping...")
            continue
            
        jpg_files = glob.glob(os.path.join(left_eye_dir, "*.jpg"))
        if not jpg_files:
            print(f"Warning: No JPG files found in {left_eye_dir}, skipping...")
            continue
        
        # Use the leftEye directory as the image directory
        image_dir = left_eye_dir
        
        datasets.append((folder_path, csv_file, image_dir))
    
    return datasets


def batch_convert_vr_datasets(input_dir, output_dir, episode_length=None, update_constants=False, task_name="vr_dataset", ignore_devices=None):
    """
    Process multiple VR datasets in batch mode, combining them as episodes of a single task.
    
    Args:
        input_dir: Path to the parent directory containing dataset folders
        output_dir: Directory to save HDF5 files (all datasets go to the same directory)
        episode_length: Optional length to split episodes
        update_constants: Whether to update constants.py with the new task
        task_name: Name for the combined task in constants.py
        ignore_devices: List of devices to ignore (e.g., ['left_controller', 'right_controller', 'hmd'])
    """
    # Find all dataset folders
    print(f"Scanning for datasets in {input_dir}...")
    datasets = find_dataset_folders(input_dir)
    
    if not datasets:
        print("No valid datasets found. Each dataset should have a CSV file and a subfolder with JPG images.")
        return
        
    print(f"Found {len(datasets)} valid datasets to process")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the current episode count in the output directory
    existing_episodes = [f for f in os.listdir(output_dir) if f.startswith('episode_') and f.endswith('.hdf5')]
    start_episode_idx = len(existing_episodes)
    
    # Process each dataset
    total_episodes = 0
    for i, (folder_path, csv_file, image_dir) in enumerate(tqdm(datasets, desc="Processing datasets")):
        folder_name = os.path.basename(folder_path)
        
        print(f"\nProcessing dataset {i+1}/{len(datasets)}: {folder_name}")
        print(f"  CSV file: {csv_file}")
        print(f"  Image directory: {image_dir}")
        print(f"  Output directory: {output_dir}")
        print(f"  Starting from episode index: {start_episode_idx + total_episodes}")
        
        # Reset the base_time attribute in find_closest_image function before processing a new dataset
        from convert_new_vr_dataset import find_closest_image
        if hasattr(find_closest_image, "base_time"):
            delattr(find_closest_image, "base_time")
        
        # Convert the dataset
        num_new_episodes = convert_vr_dataset_to_hdf5(
            csv_file,
            image_dir,
            output_dir,
            episode_length,
            start_episode_idx=start_episode_idx + total_episodes,
            ignore_devices=ignore_devices
        )
        
        total_episodes += num_new_episodes
    
    print(f"\nAll datasets processed successfully. Created {total_episodes} episodes in {output_dir}")
    
    # Update constants.py if requested
    if update_constants and total_episodes > 0:
        # Get maximum episode length from all episodes
        import h5py
        max_episode_len = 0
        
        # Iterate through all episodes to find the maximum length
        for ep_idx in range(start_episode_idx, start_episode_idx + total_episodes):
            episode_path = os.path.join(output_dir, f'episode_{ep_idx}.hdf5')
            try:
                with h5py.File(episode_path, 'r') as f:
                    current_len = f['/action'].shape[0]
                    max_episode_len = max(max_episode_len, current_len)
            except Exception as e:
                print(f"Warning: Could not read episode {ep_idx}: {e}")
                continue
        
        if max_episode_len == 0:
            print("Warning: Could not determine maximum episode length, using default value of 100")
            max_episode_len = 100
        else:
            print(f"Maximum episode length across all episodes: {max_episode_len}")
        
        # Update constants.py with the maximum episode length
        update_constants_file(output_dir, task_name, total_episodes, max_episode_len)
        
        print(f"\nTo train with this dataset, run:")
        print(f"python imitate_episodes.py --ckpt_dir ./ckpts/{task_name} --policy_class ACT --task_name {task_name} --batch_size 16 --seed 0 --num_epochs 500 --lr 1e-5 --kl_weight 5 --chunk_size 64 --hidden_dim 512 --dim_feedforward 2048")
    
    print(f"\nAll datasets processed successfully. Output saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Batch convert multiple VR datasets to HDF5 format for ACT training')
    parser.add_argument('--input_dir', type=str, required=True, help='Parent directory containing dataset folders')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save HDF5 files')
    parser.add_argument('--episode_length', type=int, default=None, help='Length of each episode (optional)')
    parser.add_argument('--update_constants', action='store_true', help='Update constants.py with new tasks')
    parser.add_argument('--task_name', type=str, default='vr_dataset', help='Name for the combined task in constants.py')
    parser.add_argument('--constants_file', type=str, default=None, help='Path to constants.py file (optional)')
    parser.add_argument('--ignore_devices', type=str, nargs='+', default=None, 
                        help='Devices to ignore (zero out). Options: hmd, left_controller, right_controller. Example: --ignore_devices left_controller')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process all datasets
    batch_convert_vr_datasets(
        args.input_dir,
        args.output_dir,
        args.episode_length,
        args.update_constants,
        args.task_name,
        ignore_devices=args.ignore_devices
    )


if __name__ == "__main__":
    main()
