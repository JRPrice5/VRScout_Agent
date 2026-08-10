#!/usr/bin/env python
"""
VR Dataset Converter for ACT Training

This script converts VR tracking data (CSV) and images to the HDF5 format
required by the ACT training framework.

Usage:
    python convert_vr_dataset.py --csv_file path/to/data.csv --image_dir path/to/images --output_dir path/to/output --episode_length 100
"""

import math
import os
import h5py
import numpy as np
import pandas as pd
import argparse
from PIL import Image
from tqdm import tqdm
import numpy as np
from pyquaternion import Quaternion
from pathlib import Path


def calculate_quaternion_delta(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Calculate the delta quaternion that rotates from q1 to q2.

    Both q1 and q2 are arrays in [x, y, z, w] format.

    Returns:
        A numpy array [x, y, z, w] representing the rotation delta.
    """
    # Convert input arrays [x, y, z, w] → pyquaternion’s (w, x, y, z)
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


def convert_vr_dataset_to_hdf5(
    csv_file, image_dir, output_dir, episode_length=None, start_episode_idx=0
):
    """
    Convert VR dataset to HDF5 format for ACT training.

    Args:
        csv_file: Path to CSV file with VR data
        image_dir: Directory containing images (sN.jpg)
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
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} records")

    # Determine episode splits
    if episode_length is None:
        # Treat entire dataset as one episode
        episode_indices = [(0, len(df))]
    else:
        # Split into episodes of specified length
        episode_indices = [
            (i, min(i + episode_length, len(df)))
            for i in range(0, len(df), episode_length)
        ]

    print(f"Converting data into {len(episode_indices)} episodes...")

    # Process each episode
    for i, (start_idx, end_idx) in enumerate(tqdm(episode_indices)):
        episode_idx = start_episode_idx + i
        episode_df = df.iloc[start_idx:end_idx]

        # Create HDF5 file for this episode
        output_file = os.path.join(output_dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(output_file, "w") as f:
            # Define qpos columns (all state variables)
            qpos_columns = [
                # Head (local position)
                "head_pos_x",
                "head_pos_y",
                "head_pos_z",
                "head_dir_x",
                "head_dir_y",
                "head_dir_z",
                "head_dir_w",
                # Left controller (local position)
                "left_controller_pos_x",
                "left_controller_pos_y",
                "left_controller_pos_z",
                "left_controller_dir_x",
                "left_controller_dir_y",
                "left_controller_dir_z",
                "left_controller_dir_w",
                # Right controller (local position)
                "right_controller_pos_x",
                "right_controller_pos_y",
                "right_controller_pos_z",
                "right_controller_dir_x",
                "right_controller_dir_y",
                "right_controller_dir_z",
                "right_controller_dir_w",
                # XR Origin (world position)
                "xr_origin_pos_x",
                "xr_origin_pos_y",
                "xr_origin_pos_z",
            ]

            # Store qpos data
            qpos_data = episode_df[qpos_columns].values
            f.create_dataset("/observations/qpos", data=qpos_data)

            # Create dummy qvel (velocity) data
            qvel_data = np.zeros((len(episode_df), len(qpos_columns)))
            f.create_dataset("/observations/qvel", data=qvel_data)

            # Create action data (using next row values directly instead of deltas)
            action_data = []
            for i in range(len(episode_df) - 1):
                next_row = episode_df.iloc[i + 1]

                # Head position and rotation from next row
                head_pos = [
                    next_row["head_pos_x"],
                    next_row["head_pos_y"],
                    next_row["head_pos_z"],
                ]

                head_rot = [
                    next_row["head_dir_x"],
                    next_row["head_dir_y"],
                    next_row["head_dir_z"],
                    next_row["head_dir_w"],
                ]

                # Left controller position and rotation from next row
                left_pos = [
                    next_row["left_controller_pos_x"],
                    next_row["left_controller_pos_y"],
                    next_row["left_controller_pos_z"],
                ]

                left_rot = [
                    next_row["left_controller_dir_x"],
                    next_row["left_controller_dir_y"],
                    next_row["left_controller_dir_z"],
                    next_row["left_controller_dir_w"],
                ]

                # Right controller position and rotation from next row
                right_pos = [
                    next_row["right_controller_pos_x"],
                    next_row["right_controller_pos_y"],
                    next_row["right_controller_pos_z"],
                ]

                right_rot = [
                    next_row["right_controller_dir_x"],
                    next_row["right_controller_dir_y"],
                    next_row["right_controller_dir_z"],
                    next_row["right_controller_dir_w"],
                ]

                # XR Origin position from next row
                xr_origin_pos = [
                    next_row["xr_origin_pos_x"],
                    next_row["xr_origin_pos_y"],
                    next_row["xr_origin_pos_z"],
                ]

                # Combine all next state values into action vector
                action = (
                    head_pos
                    + head_rot
                    + left_pos
                    + left_rot
                    + right_pos
                    + right_rot
                    + xr_origin_pos
                )

                action_data.append(action)

            # Store action data
            action_data = np.array(action_data)
            f.create_dataset("/action", data=action_data)

            # Store images
            images_group = f.create_group("/observations/images")

            # Load and store each image
            image_data = []
            for i, row in episode_df.iterrows():
                key = int(row["key"])
                img_path = os.path.join(image_dir, f"s{key}.jpg")

                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    img_array = np.array(img)
                    image_data.append(img_array)
                else:
                    print(f"Warning: Image {img_path} not found")
                    # Create a blank image with the same dimensions as others
                    # If this is the first image, use a default size
                    if image_data:
                        blank_img = np.zeros_like(image_data[0])
                    else:
                        blank_img = np.zeros((224, 224, 3), dtype=np.uint8)
                    image_data.append(blank_img)

            # Store all images directly under the camera name as a dataset
            if image_data:
                # Convert to numpy array with shape (num_frames, height, width, channels)
                image_array = np.array(image_data)
                # Create dataset with camera name directly
                images_group.create_dataset("camera1", data=image_array)
            else:
                print("Warning: No images found")
                # Create empty dataset
                images_group.create_dataset(
                    "camera1",
                    data=np.zeros((len(episode_df), 224, 224, 3), dtype=np.uint8),
                )

            # Add attribute to indicate this is real data (not simulation)
            f.attrs["sim"] = False

    print(f"Converted {len(episode_indices)} episodes to HDF5 format in {output_dir}")
    return len(episode_indices)


def update_constants_file(
    output_dir, task_name, num_episodes, episode_len, constants_file=None
):
    """
    Update the constants.py file to include the new task configuration.

    Args:
        output_dir: Directory where HDF5 files are saved
        task_name: Name for the new task
        num_episodes: Number of episodes
        episode_len: Length of each episode
        constants_file: Path to constants.py file (optional)
    """
    if constants_file is None:
        constants_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "constants.py"
        )

    if os.path.exists(constants_file):
        with open(constants_file, "r") as f:
            content = f.read()

        # Check if task already exists
        if f"'{task_name}':" in content:
            print(f"Task '{task_name}' already exists in constants.py")
            return

        # Find the SIM_TASK_CONFIGS dictionary
        if "SIM_TASK_CONFIGS = {" in content:
            # Try to get relative path from DATA_DIR
            try:
                rel_path = os.path.relpath(output_dir, os.path.dirname(constants_file))
                dataset_dir_path = f"DATA_DIR + '/{rel_path.replace(os.sep, '/')}'"
            except ValueError:
                # If on different drives, use absolute path
                dataset_dir_path = f"'{output_dir}'"  # Use absolute path

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
            start_idx = content.find("SIM_TASK_CONFIGS = {")
            if start_idx == -1:
                print("Could not find SIM_TASK_CONFIGS dictionary")
                return

            # Count braces to find the end of the dictionary
            brace_count = 1  # We start after the opening brace
            pos = content.find("{", start_idx) + 1
            while brace_count > 0 and pos < len(content):
                if content[pos] == "{":
                    brace_count += 1
                elif content[pos] == "}":
                    brace_count -= 1
                pos += 1

            if brace_count != 0:
                print(
                    "Could not properly parse the SIM_TASK_CONFIGS dictionary structure"
                )
                return

            # pos is now just after the closing brace of SIM_TASK_CONFIGS
            # Go back one character to insert before the closing brace
            end_idx = pos - 1
            updated_content = content[:end_idx] + new_task + content[end_idx:]

            with open(constants_file, "w") as f:
                f.write(updated_content)

            print(f"Added task '{task_name}' to constants.py")
        else:
            print("Could not find SIM_TASK_CONFIGS in constants.py")
    else:
        print(f"Warning: constants.py not found at {constants_file}")


# TODO: Finish shard to dataset conversion
# TODO: determine whether this need optimisation such as concurrent, parallel or batch processing for large datasets
def convert_shard_to_hdf5(
    replay_index, shard_dir, output_dir, episode_length=None, start_episode_idx=0
):
    """
    Function may raise a TypeError
    Convert DzRamen BeatLeader downloads to HDF5 format for ACT training.


        replay_index: Index of shards to convert
        shard_dir: Directory containing replay shards (.npy)
        output_dir: Directory to save HDF5 files
        episode_length: Optional length to split episodes
        start_episode_idx: Starting index for episode numbering (for batch processing)

    Returns:
        Number of episodes created
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load shard data
    # x1 Extra Stats
    # shape: (frames/(2*frameCountOutput), frameCountOutput+6)
    # details: frameCountOutput timestamps for the future frames, njs / 30*timeScale, jd / 30, live replay height, pm, sc, od

    # x2 Notes and bombs
    # shape: (frames/(2*frameCountOutput), frameCountOutput, 31)
    # details: note&bomb times and positions

    # x3 Walls,
    # shape: (frames/(2*frameCountOutput), frameCountOutput, 6)
    # details: wall times and positions

    # y - Future Frames (actions)
    # shape: (frames/(2*frameCountOutput), frameCountOutput, 21)
    # details: future qpos

    print(f"Loading data from replay {replay_index}...")
    shard = []
    for shard_type in ["x1", "x2", "x3", "y"]:
        filename = f"{shard_type}_{replay_index}"
        filepath = os.path.join(shard_dir, filename)
        shard.append(np.load(filepath, allow_pickle=True))

    # Shard: list(np.ndarray)

    # Convert to dataframe

    print(f"Loaded {len(df)} records")

    # Determine episode splits
    if episode_length is None:
        # Treat entire dataset as one episode
        episode_indices = [(0, len(df))]
    else:
        # Split into episodes of specified length
        episode_indices = [
            (i, min(i + episode_length, len(df)))
            for i in range(0, len(df), episode_length)
        ]

    print(f"Converting data into {len(episode_indices)} episodes...")

    # Process each episode
    for i, (start_idx, end_idx) in enumerate(tqdm(episode_indices)):
        episode_idx = start_episode_idx + i
        episode_df = df.iloc[start_idx:end_idx]

        # Create HDF5 file for this episode
        output_file = os.path.join(output_dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(output_file, "w") as f:
            # Define qpos columns (all state variables)
            qpos_columns = [
                # Head (local position)
                "head_pos_x",
                "head_pos_y",
                "head_pos_z",
                "head_dir_x",
                "head_dir_y",
                "head_dir_z",
                "head_dir_w",
                # Left controller (local position)
                "left_controller_pos_x",
                "left_controller_pos_y",
                "left_controller_pos_z",
                "left_controller_dir_x",
                "left_controller_dir_y",
                "left_controller_dir_z",
                "left_controller_dir_w",
                # Right controller (local position)
                "right_controller_pos_x",
                "right_controller_pos_y",
                "right_controller_pos_z",
                "right_controller_dir_x",
                "right_controller_dir_y",
                "right_controller_dir_z",
                "right_controller_dir_w",
                # XR Origin (world position)
                "xr_origin_pos_x",
                "xr_origin_pos_y",
                "xr_origin_pos_z",
                # Controller inputs
                "IndexTrigger_0",
                "IndexTrigger_1",
                "HandTrigger_0",
                "HandTrigger_1",
                "Thumbstick_0_x",
                "Thumbstick_0_y",
                "Thumbstick_1_x",
                "Thumbstick_1_y",
            ]

            # Store qpos data
            qpos_data = episode_df[qpos_columns].values
            f.create_dataset("/observations/qpos", data=qpos_data)

            # Create dummy qvel (velocity) data
            qvel_data = np.zeros((len(episode_df), len(qpos_columns)))
            f.create_dataset("/observations/qvel", data=qvel_data)

            # Create action data (using next row values directly instead of deltas)
            action_data = []
            for i in range(len(episode_df) - 1):
                next_row = episode_df.iloc[i + 1]

                # Head position and rotation from next row
                head_pos = [
                    next_row["head_pos_x"],
                    next_row["head_pos_y"],
                    next_row["head_pos_z"],
                ]

                head_rot = [
                    next_row["head_dir_x"],
                    next_row["head_dir_y"],
                    next_row["head_dir_z"],
                    next_row["head_dir_w"],
                ]

                # Left controller position and rotation from next row
                left_pos = [
                    next_row["left_controller_pos_x"],
                    next_row["left_controller_pos_y"],
                    next_row["left_controller_pos_z"],
                ]

                left_rot = [
                    next_row["left_controller_dir_x"],
                    next_row["left_controller_dir_y"],
                    next_row["left_controller_dir_z"],
                    next_row["left_controller_dir_w"],
                ]

                # Right controller position and rotation from next row
                right_pos = [
                    next_row["right_controller_pos_x"],
                    next_row["right_controller_pos_y"],
                    next_row["right_controller_pos_z"],
                ]

                right_rot = [
                    next_row["right_controller_dir_x"],
                    next_row["right_controller_dir_y"],
                    next_row["right_controller_dir_z"],
                    next_row["right_controller_dir_w"],
                ]

                # XR Origin position from next row
                xr_origin_pos = [
                    next_row["xr_origin_pos_x"],
                    next_row["xr_origin_pos_y"],
                    next_row["xr_origin_pos_z"],
                ]

                # Controller inputs from next row
                controller_inputs = [
                    next_row["IndexTrigger_0"],
                    next_row["IndexTrigger_1"],
                    next_row["HandTrigger_0"],
                    next_row["HandTrigger_1"],
                    next_row["Thumbstick_0_x"],
                    next_row["Thumbstick_0_y"],
                    next_row["Thumbstick_1_x"],
                    next_row["Thumbstick_1_y"],
                ]

                # Combine all next state values into action vector
                action = (
                    head_pos
                    + head_rot
                    + left_pos
                    + left_rot
                    + right_pos
                    + right_rot
                    + xr_origin_pos
                    + controller_inputs
                )

                action_data.append(action)

            # Store action data
            action_data = np.array(action_data)
            f.create_dataset("/action", data=action_data)

            # Store images
            images_group = f.create_group("/observations/images")

            # Load and store each image
            image_data = []
            for i, row in episode_df.iterrows():
                key = int(row["key"])
                img_path = os.path.join(image_dir, f"s{key}.jpg")

                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    img_array = np.array(img)
                    image_data.append(img_array)
                else:
                    print(f"Warning: Image {img_path} not found")
                    # Create a blank image with the same dimensions as others
                    # If this is the first image, use a default size
                    if image_data:
                        blank_img = np.zeros_like(image_data[0])
                    else:
                        blank_img = np.zeros((224, 224, 3), dtype=np.uint8)
                    image_data.append(blank_img)

            # Store all images directly under the camera name as a dataset
            if image_data:
                # Convert to numpy array with shape (num_frames, height, width, channels)
                image_array = np.array(image_data)
                # Create dataset with camera name directly
                images_group.create_dataset("camera1", data=image_array)
            else:
                print("Warning: No images found")
                # Create empty dataset
                images_group.create_dataset(
                    "camera1",
                    data=np.zeros((len(episode_df), 224, 224, 3), dtype=np.uint8),
                )

            # Add attribute to indicate this is real data (not simulation)
            f.attrs["sim"] = False

    print(f"Converted {len(episode_indices)} episodes to HDF5 format in {output_dir}")
    return len(episode_indices)


# TODO: adjust for frames format (hpos, hrot, leftpos, leftrot, rightpos, rightrot)
# TODO Optimise past brute force, considering that the next frames future frames only change by the first and last frame
def _interpolate_actions(
        self, input_frames: np.ndarray, timestamps: np.ndarray, interpolation_rate: int
) -> np.ndarray:
    """
        Interpolates variable length action frames into fixed length action frames for ACT input requirements

        Args:
            input_frames: 3D poses sampled at a variable rate. Shape (2*frameCountOutput, frameCountOutput, 21)
            timestamps: when each pose was sampled. Shape (2*frameCountOutput, frameCountOutput)
            interpolation_rate: self explanatory

        Returns:
            ndarray of input_frames, interpolated at interpolation_rate. Shape (2*frameCountOutput, frameCountOutput, 21)

    """
    output_frames = np.zeros(input_frames.shape)
    replay_index = 0
    for i in range(output_frames.shape[0])  # For each frame
        for j in range(output_frames.shape[1]): # For each future frame
            interpolate_time = (j + 1) * (1 / interpolation_rate)

            replay_time = timestamps[j]
            while replay_time < interpolate_time:
                replay_index += 1
                replay_time = timestamps[

            output_frames[i] = self._interpolate_frame(
                time=interpolate_time,
                pre_index=replay_index - 1,
                post_index=replay_index,
                frames=input_frames,
            )
    return output_frames


# TODO: adjust for frames format (hpos, hrot, leftpos, leftrot, rightpos, rightrot)
# TODO: refactor body into interpolate position and interpolate rotation
def _interpolate_frame(
    self, time: float, pre_index: int, post_index: int, frames: np.ndarray
) -> np.ndarray:
    pre_frame = frames[pre_index]
    pre_time = pre_frame[0]
    post_frame = frames[post_index]
    post_time = post_frame[0]

    # Take keyPress from nearest frame
    pre_time_delta = time - pre_time
    post_time_delta = post_time - time
    nearest_frame = post_frame if pre_time_delta > post_time_delta else pre_frame
    key_press = nearest_frame[3]

    # Calculate interpolation ratio
    interpolation_ratio = pre_time_delta / (post_time - pre_time)

    # Interpolate x position
    pre_x = pre_frame[1]
    post_x = post_frame[1]
    x_interpolated = pre_x + interpolation_ratio * (post_x - pre_x)

    # Interpolate y position
    pre_y = pre_frame[2]
    post_y = post_frame[2]
    y_interpolated = pre_y + interpolation_ratio * (post_y - pre_y)

    return np.array([time, x_interpolated, y_interpolated, key_press])


def main():
    parser = argparse.ArgumentParser(
        description="Convert VR dataset to HDF5 format for ACT training"
    )
    parser.add_argument(
        "--csv_file", type=str, required=True, help="Path to CSV file with VR data"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing images (sN.jpg)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save HDF5 files"
    )
    parser.add_argument(
        "--episode_length",
        type=int,
        default=None,
        help="Length of each episode (optional)",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="vr_dataset",
        help="Name for the task in constants.py",
    )
    parser.add_argument(
        "--update_constants",
        action="store_true",
        help="Update constants.py with new task",
    )
    parser.add_argument(
        "--constants_file",
        type=str,
        default=None,
        help="Path to constants.py file (optional)",
    )

    args = parser.parse_args()

    # Convert dataset
    num_episodes = convert_vr_dataset_to_hdf5(
        args.csv_file, args.image_dir, args.output_dir, args.episode_length
    )

    # Count number of episodes
    num_episodes = len(
        [
            f
            for f in os.listdir(args.output_dir)
            if f.startswith("episode_") and f.endswith(".hdf5")
        ]
    )

    # Get episode length from first episode
    if num_episodes > 0:
        first_episode = os.path.join(args.output_dir, f"episode_0.hdf5")
        with h5py.File(first_episode, "r") as f:
            episode_len = f["/action"].shape[0]
    else:
        episode_len = args.episode_length or 100

    # Update constants.py if requested
    if args.update_constants:
        update_constants_file(
            args.output_dir,
            args.task_name,
            num_episodes,
            episode_len,
            args.constants_file,
        )
        print(f"\nTo train with this dataset, run:")
        print(
            f"python imitate_episodes.py --ckpt_dir ./ckpts/{args.task_name} --policy_class ACT --task_name {args.task_name} --batch_size 16 --seed 0 --num_epochs 500 --lr 1e-5 --kl_weight 5 --chunk_size 64 --hidden_dim 512 --dim_feedforward 2048"
        )


if __name__ == "__main__":
    main()
