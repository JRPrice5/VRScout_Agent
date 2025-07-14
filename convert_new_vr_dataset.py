#!/usr/bin/env python
"""
New VR Dataset Converter for ACT Training

This script converts VR tracking data (CSV) and images to the HDF5 format
required by the ACT training framework.

The script handles the new CSV format with columns:
- timestamp
- hmd_position_x/y/z, hmd_rotation_x/y/z/w
- controller_lefthand_position_x/y/z, controller_lefthand_rotation_x/y/z/w
- controller_righthand_position_x/y/z, controller_righthand_rotation_x/y/z/w
- lefthand_trigger_state_changed, lefthand_grip_state_changed
- righthand_trigger_state_changed, righthand_grip_state_changed

And images named with timestamps: YYYY-MM-DD-HH-MM-SS-mmm.jpg

Usage:
    python convert_new_vr_dataset.py --csv_file path/to/data.csv --image_dir path/to/images --output_dir path/to/output --episode_length 100
"""

import os
import h5py
import numpy as np
import pandas as pd
import argparse
from PIL import Image
from tqdm import tqdm
from pyquaternion import Quaternion
from datetime import datetime
import matplotlib.pyplot as plt
import re


def calculate_quaternion_delta(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Calculate the delta quaternion that rotates from q1 to q2.

    Both q1 and q2 are arrays in [x, y, z, w] format.

    Returns:
        A numpy array [x, y, z, w] representing the rotation delta.
    """
    # Convert input arrays [x, y, z, w] → pyquaternion's (w, x, y, z)
    q1_py = Quaternion(w=q1[3], x=q1[0], y=q1[1], z=q1[2]).normalised
    q2_py = Quaternion(w=q2[3], x=q2[0], y=q2[1], z=q2[2]).normalised

    # Enforce shortest path: if dot < 0, flip one quaternion
    if np.dot(q1_py.elements, q2_py.elements) < 0:
        q2_py = -q2_py

    # Delta is the rotation that takes q1 → q2
    delta = q2_py * q1_py.inverse
    delta = delta.normalised

    # pyquaternion stores elements as [w, x, y, z]; reorder for [x, y, z, w]
    return np.array([delta.x, delta.y, delta.z, delta.w])


def timestamp_to_seconds(timestamp_str):
    """
    Convert timestamp string to seconds from epoch.
    Used for matching images with CSV rows.
    
    Args:
        timestamp_str: Timestamp string in format YYYY-MM-DD-HH-MM-SS-mmm
        
    Returns:
        Timestamp in seconds with millisecond precision
    """
    # Parse the timestamp string
    match = re.match(r'(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{3})', timestamp_str)
    if match:
        year, month, day, hour, minute, second, millisecond = map(int, match.groups())
        dt = datetime(year, month, day, hour, minute, second, millisecond * 1000)
        return dt.timestamp()
    return None


def find_closest_image(timestamp, image_timestamps):
    """
    Find the closest image timestamp to the given CSV row timestamp.
    
    Args:
        timestamp: CSV row timestamp in seconds from start of recording
        image_timestamps: Dictionary of {timestamp_seconds: image_path}
        
    Returns:
        Path to the closest matching image
    """
    if not image_timestamps:
        return None
    
    # Get the first timestamp from images to use as reference (base time)
    if not hasattr(find_closest_image, "base_time"):
        find_closest_image.base_time = min(image_timestamps.keys())
    
    # Find the closest timestamp by comparing relative times
    # CSV timestamp is seconds from start, so add it to the base image time
    target_time = find_closest_image.base_time + float(timestamp)
    
    # More efficient implementation that doesn't create a list of all differences
    min_diff = float('inf')
    closest_ts = None
    
    for ts in image_timestamps.keys():
        diff = abs(ts - target_time)
        if diff < min_diff:
            min_diff = diff
            closest_ts = ts
    
    # If the difference is too large (more than 1 second), consider it a mismatch
    if min_diff > 1.0:
        print(f"Warning: Closest image timestamp differs by {min_diff:.3f} seconds")
        
    return image_timestamps[closest_ts]


def convert_vr_dataset_to_hdf5(csv_file, image_dir, output_dir, episode_length=None, start_episode_idx=0):
    """
    Convert VR dataset to HDF5 format for ACT training.
    
    Args:
        csv_file: Path to CSV file with VR data
        image_dir: Directory containing images (timestamp format)
        output_dir: Directory to save HDF5 files
        episode_length: Optional length to split episodes
        start_episode_idx: Starting index for episode numbering (for batch processing)
        
    Returns:
        Number of episodes created
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CSV data
    print(f"Loading data from {csv_file}...")
    
    # Step 1: Read header and count rows to determine episode structure first
    with open(csv_file, encoding='utf-8') as f:
        header = f.readline().strip()
        expected_col_count = len(header.split(','))
        row_count = sum(1 for _ in f) + 1  # +1 for header
    
    print(f"Found {row_count-1} records in CSV file")
    
    # Determine episode splits before loading full data
    if episode_length is None:
        # Treat entire dataset as one episode
        episode_indices = [(0, row_count-1)]  # -1 to exclude header
    else:
        # Split into episodes of specified length
        episode_indices = [(i, min(i + episode_length, row_count-1)) 
                          for i in range(1, row_count, episode_length)]  # Start at 1 to skip header
    
    print(f"Will convert data into {len(episode_indices)} episodes...")
    
    # We'll process the CSV in chunks by episode to save memory
    header_fields = header.split(',')
    
    # Get column indices for required fields to avoid loading entire dataframe
    required_cols = [
        'timestamp',
        'hmd_position_x', 'hmd_position_y', 'hmd_position_z',
        'hmd_rotation_x', 'hmd_rotation_y', 'hmd_rotation_z', 'hmd_rotation_w',
        'controller_lefthand_position_x', 'controller_lefthand_position_y', 'controller_lefthand_position_z',
        'controller_lefthand_rotation_x', 'controller_lefthand_rotation_y', 'controller_lefthand_rotation_z', 'controller_lefthand_rotation_w',
        'controller_righthand_position_x', 'controller_righthand_position_y', 'controller_righthand_position_z',
        'controller_righthand_rotation_x', 'controller_righthand_rotation_y', 'controller_righthand_rotation_z', 'controller_righthand_rotation_w',
        'lefthand_trigger_state_changed', 'lefthand_grip_state_changed',
        'righthand_trigger_state_changed', 'righthand_grip_state_changed'
    ]
    
    # Process image directory - map timestamps to file paths
    print(f"Processing images from {image_dir}...")
    image_timestamps = {}
    image_count = 0
    
    # Only process image directory if it exists
    if os.path.exists(image_dir):
        # Use a generator to avoid loading all filenames into memory at once
        def get_image_files():
            for filename in os.listdir(image_dir):
                if filename.endswith('.jpg'):
                    yield filename
        
        # Process images in batches to reduce memory usage
        batch_size = 100
        image_files = get_image_files()
        processing = True
        
        while processing:
            batch = []
            # Get next batch
            for _ in range(batch_size):
                try:
                    batch.append(next(image_files))
                except StopIteration:
                    processing = False
                    break
            
            # Process this batch
            for filename in batch:
                # Extract timestamp from filename (without extension)
                base_name = os.path.splitext(filename)[0]
                ts_seconds = timestamp_to_seconds(base_name)
                if ts_seconds:
                    image_timestamps[ts_seconds] = os.path.join(image_dir, filename)
                    image_count += 1
    
    print(f"Found {len(image_timestamps)} images with valid timestamps")
    
    # Initialize state tracking for trigger and grip
    lefthand_trigger_state = 0  # Initial state is FALSE (0)
    lefthand_grip_state = 0     # Initial state is FALSE (0)
    righthand_trigger_state = 0 # Initial state is FALSE (0)
    righthand_grip_state = 0    # Initial state is FALSE (0)
    
    # Process each episode
    for i, (start_idx, end_idx) in enumerate(tqdm(episode_indices)):
        episode_idx = start_episode_idx + i
        
        # Read only the chunk of CSV data needed for this episode
        episode_data = []
        col_indices = {col: header_fields.index(col) for col in required_cols if col in header_fields}
        
        with open(csv_file, encoding='utf-8') as f:
            # Skip the header row first
            next(f)
            
            # Skip to the start of this episode
            for _ in range(start_idx):
                next(f)
            
            # Read lines for this episode
            for _ in range(end_idx - start_idx):
                try:
                    line = next(f)
                    fields = line.rstrip('\n').split(',')
                    
                    # Ensure we have enough fields
                    if len(fields) < expected_col_count:
                        fields += [''] * (expected_col_count - len(fields))
                    elif len(fields) > expected_col_count:
                        fields = fields[:expected_col_count]
                    
                    # Extract only the required fields
                    row_data = {col: fields[idx] for col, idx in col_indices.items()}
                    episode_data.append(row_data)
                except StopIteration:
                    break
        
        # Convert to DataFrame only for this episode
        episode_df = pd.DataFrame(episode_data)
        
        # Create HDF5 file for this episode
        output_file = os.path.join(output_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(output_file, 'w') as f:
            # Prepare data arrays for this episode
            qpos_data = []
            action_data = []
            image_data = []
            
            # Reset state tracking at the beginning of each episode
            lefthand_trigger_state = 0
            lefthand_grip_state = 0
            righthand_trigger_state = 0
            righthand_grip_state = 0
            
            # Process each row in the episode
            for idx, row in episode_df.iterrows():
                # Update trigger and grip states based on state change columns
                if row['lefthand_trigger_state_changed'] != "":
                    lefthand_trigger_state = 1.0 if row['lefthand_trigger_state_changed'].lower() == 'true' else 0.0
                
                if row['lefthand_grip_state_changed'] != "":
                    lefthand_grip_state = 1.0 if row['lefthand_grip_state_changed'].lower() == 'true' else 0.0
                
                if row['righthand_trigger_state_changed'] != "":
                    righthand_trigger_state = 1.0 if row['righthand_trigger_state_changed'].lower() == 'true' else 0.0
                
                if row['righthand_grip_state_changed'] != "":
                    righthand_grip_state = 1.0 if row['righthand_grip_state_changed'].lower() == 'true' else 0.0
                        
                # Map new CSV columns to qpos format
                qpos_row = [
                    # Head (local position)
                    float(row['hmd_position_x']), float(row['hmd_position_y']), float(row['hmd_position_z']),
                    float(row['hmd_rotation_x']), float(row['hmd_rotation_y']), float(row['hmd_rotation_z']), float(row['hmd_rotation_w']),
                    
                    # Left controller (local position)
                    float(row['controller_lefthand_position_x']), float(row['controller_lefthand_position_y']), float(row['controller_lefthand_position_z']),
                    float(row['controller_lefthand_rotation_x']), float(row['controller_lefthand_rotation_y']), float(row['controller_lefthand_rotation_z']), float(row['controller_lefthand_rotation_w']),
                    
                    # Right controller (local position)
                    float(row['controller_righthand_position_x']), float(row['controller_righthand_position_y']), float(row['controller_righthand_position_z']),
                    float(row['controller_righthand_rotation_x']), float(row['controller_righthand_rotation_y']), float(row['controller_righthand_rotation_z']), float(row['controller_righthand_rotation_w']),
                    
                    # XR Origin (world position) - assuming 0 for now as it's not in the new CSV
                    0.0, 0.0, 0.0,
                    
                    # Controller inputs - map trigger and grip states to expected format
                    lefthand_trigger_state, righthand_trigger_state,
                    lefthand_grip_state, righthand_grip_state,
                    0.0, 0.0,  # Thumbstick_0_x/y - assuming 0 as not in new CSV
                    0.0, 0.0   # Thumbstick_1_x/y - assuming 0 as not in new CSV
                ]
                
                qpos_data.append(qpos_row)
                
                # Find the closest image for this timestamp
                if image_timestamps:
                    closest_image_path = find_closest_image(float(row['timestamp']), image_timestamps)
                    if closest_image_path and os.path.exists(closest_image_path):
                        # Load image and convert to array more efficiently
                        with Image.open(closest_image_path) as img:
                            # Fix inverted image by flipping vertically only (horizontal flip was not needed)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM)
                            
                            # Convert to numpy array with correct dtype
                            img_array = np.array(img, dtype=np.uint8)
                            image_data.append(img_array)
                            
                            # Debug: Plot the first converted image
                            if len(image_data) == 1:
                                plt.figure(figsize=(10, 8))
                                plt.imshow(img_array)
                                plt.title(f"First converted image (corrected): {os.path.basename(closest_image_path)}")
                                plt.axis('on')
                                debug_output_path = os.path.join(os.path.dirname(output_file), 'debug_first_image.png')
                                plt.savefig(debug_output_path)
                                plt.close()
                                print(f"Debug image saved to: {debug_output_path}")
                    else:
                        # Create a blank image if no match found
                        if image_data:
                            blank_img = np.zeros_like(image_data[0])
                        else:
                            blank_img = np.zeros((224, 224, 3), dtype=np.uint8)
                        image_data.append(blank_img)
                        print(f"Warning: No matching image found for timestamp {row['timestamp']}")
            
            # Create action data (using next row values directly)
            for i in range(len(qpos_data) - 1):
                action_data.append(qpos_data[i + 1])
            
            # Convert lists to numpy arrays
            qpos_data = np.array(qpos_data)
            action_data = np.array(action_data)
            
            # Store qpos data with compression
            f.create_dataset('/observations/qpos', data=qpos_data, compression='gzip', compression_opts=4)
            
            # Create dummy qvel (velocity) data with compression
            qvel_data = np.zeros((len(qpos_data), qpos_data.shape[1]), dtype=np.float32)
            f.create_dataset('/observations/qvel', data=qvel_data, compression='gzip', compression_opts=4)
            
            # Store action data with compression
            f.create_dataset('/action', data=action_data, compression='gzip', compression_opts=4)
            
            # Store images
            images_group = f.create_group('/observations/images')
            
            if image_data:
                # Process images in smaller batches to reduce peak memory usage
                img_shape = image_data[0].shape
                img_dataset = images_group.create_dataset('camera1', 
                                                shape=(len(image_data), *img_shape), 
                                                dtype=np.uint8,
                                                compression='gzip',
                                                compression_opts=4,
                                                chunks=(1, *img_shape))
                
                # Write images in batches
                batch_size = 10
                for i in range(0, len(image_data), batch_size):
                    batch_end = min(i + batch_size, len(image_data))
                    img_dataset[i:batch_end] = np.array(image_data[i:batch_end])
                    
                    # Clear processed batch from memory
                    for j in range(i, batch_end):
                        image_data[j] = None
                
                # Force garbage collection after processing images
                import gc
                gc.collect()
            else:
                print("Warning: No images found")
                # Create empty dataset
                images_group.create_dataset('camera1', data=np.zeros((len(qpos_data), 224, 224, 3), dtype=np.uint8))
            
            # Add attribute to indicate this is real data (not simulation)
            f.attrs['sim'] = False
            
    print(f"Converted {len(episode_indices)} episodes to HDF5 format in {output_dir}")
    return len(episode_indices)


def update_constants_file(output_dir, task_name, num_episodes, episode_len, constants_file=None):
    """
    Update the constants.py file to include the new task configuration.
    
    Args:
        output_dir: Directory where HDF5 files are saved
        task_name: Name for the new task
        num_episodes: Number of episodes
        episode_len: Length of each episode
        constants_file: Path to constants.py file (optional)
    """
    # Find constants.py file if not specified
    if constants_file is None:
        # Try to find it in common locations
        possible_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'constants.py'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'constants.py'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                constants_file = path
                break
    
    if constants_file and os.path.exists(constants_file):
        print(f"Updating constants file: {constants_file}")
        
        # Read current content
        with open(constants_file, 'r') as f:
            content = f.read()
            
        if 'SIM_TASK_CONFIGS' in content:
            # Format the output directory path
            if os.path.isabs(output_dir):
                dataset_dir_path = f"'{output_dir}'"  # Use absolute path
            else:
                dataset_dir_path = f"'{output_dir}'"  # Use relative path
            
            # Create new task entry
            new_task = f"""    '{task_name}': {{
        'dataset_dir': r{dataset_dir_path},
        'num_episodes': {num_episodes},
        'episode_len': {episode_len},
        'camera_names': ['camera1']
    }},
"""
            
            # Find the proper position to insert the new task
            # Look for the end of the SIM_TASK_CONFIGS dictionary
            start_idx = content.find('SIM_TASK_CONFIGS = {')
            if start_idx == -1:
                print("Could not find SIM_TASK_CONFIGS dictionary")
                return
                
            # Count braces to find the end of the dictionary
            brace_count = 1  # We start after the opening brace
            pos = content.find('{', start_idx) + 1
            while brace_count > 0 and pos < len(content):
                if content[pos] == '{':
                    brace_count += 1
                elif content[pos] == '}':
                    brace_count -= 1
                pos += 1
                
            if brace_count != 0:
                print("Could not properly parse the SIM_TASK_CONFIGS dictionary structure")
                return
                
            # pos is now just after the closing brace of SIM_TASK_CONFIGS
            # Go back one character to insert before the closing brace
            end_idx = pos - 1
            updated_content = content[:end_idx] + new_task + content[end_idx:]
            
            with open(constants_file, 'w') as f:
                f.write(updated_content)
                
            print(f"Added task '{task_name}' to constants.py")
        else:
            print("Could not find SIM_TASK_CONFIGS in constants.py")
    else:
        print(f"Warning: constants.py not found at {constants_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert new VR dataset format to HDF5 format for ACT training')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file with VR data')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images (timestamp format)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save HDF5 files')
    parser.add_argument('--episode_length', type=int, default=None, help='Length of each episode (optional)')
    parser.add_argument('--task_name', type=str, default='new_vr_dataset', help='Name for the task in constants.py')
    parser.add_argument('--update_constants', action='store_true', help='Update constants.py with new task')
    parser.add_argument('--constants_file', type=str, default=None, help='Path to constants.py file (optional)')
    
    args = parser.parse_args()
    
    # Convert dataset
    num_episodes = convert_vr_dataset_to_hdf5(
        args.csv_file, 
        args.image_dir, 
        args.output_dir, 
        args.episode_length
    )
    
    # Count number of episodes
    num_episodes = len([f for f in os.listdir(args.output_dir) if f.startswith('episode_') and f.endswith('.hdf5')])
    
    # Get episode length from first episode
    if num_episodes > 0:
        first_episode = os.path.join(args.output_dir, f'episode_0.hdf5')
        with h5py.File(first_episode, 'r') as f:
            episode_len = f['/action'].shape[0]
    else:
        episode_len = args.episode_length or 100
    
    # Update constants.py if requested
    if args.update_constants:
        update_constants_file(args.output_dir, args.task_name, num_episodes, episode_len, args.constants_file)
        print(f"\nTo train with this dataset, run:")
        print(f"python imitate_episodes.py --ckpt_dir ./ckpts/{args.task_name} --policy_class ACT --task_name {args.task_name} --batch_size 16 --seed 0 --num_epochs 500 --lr 1e-5 --kl_weight 5 --chunk_size 64 --hidden_dim 512 --dim_feedforward 2048")


if __name__ == "__main__":
    main()
