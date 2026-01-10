import pathlib

### Task parameters
DATA_DIR = '.'
SIM_TASK_CONFIGS = {
    'sim_transfer_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_transfer_cube_human':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_human',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_scripted': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_human': {
        'dataset_dir': DATA_DIR + '/sim_insertion_human',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['top']
    },
    'sim_vr_task': {
        'dataset_dir': r'E:\vr-data\q3_data\HDF5',
        'num_episodes': 16,
        'episode_len': 600,
        'camera_names': ['camera1']
    },
    'sim_vr_task_2': {
        'dataset_dir': r'E:\vr-data\q3_data_2\HDF5_abs',
        'num_episodes': 20,
        'episode_len': 565,
        'camera_names': ['camera1']
    },
    'sim_vr_task_3': {
        'dataset_dir': r'E:\vr-data\BS_DATA\HDF5',
        'num_episodes': 21,
        'episode_len': 8638,
        'camera_names': ['camera1']
    },
    'sim_vr_task_sh_1': {
        'dataset_dir': r'D:\CLOVR_Output\SH_DATA_1\HDF5',
        'num_episodes': 60,
        'episode_len': 1999,
        'camera_names': ['camera1']
    },
    'sim_vr_task_6': {
        'dataset_dir': r'D:\CLOVR_Output\BS_DATA\HDF5',
        'num_episodes': 69,
        'episode_len': 11850,
        'camera_names': ['camera1']
    },
    'sim_vr_task_pw_1': {
        'dataset_dir': r'D:\CLOVR_Output\PW_FB_DATA_1\HDF5',
        'num_episodes': 20,
        'episode_len': 9083,
        'camera_names': ['camera1']
    },
    'sim_vr_task_pw_2': {
        'dataset_dir': r'D:\CLOVR_Output\PW_FB_DATA_2\HDF5',
        'num_episodes': 25,
        'episode_len': 7512,
        'camera_names': ['camera1']
    },
    'sim_vr_task_pw_3': {
        'dataset_dir': r'D:\CLOVR_Output\PW_DATA_3\HDF5',
        'num_episodes': 24,
        'episode_len': 6958,
        'camera_names': ['camera1']
    },
    'sim_vr_task_pw_4': {
        'dataset_dir': r'D:\CLOVR_Output\PW_DATA_3\HDF5',
        'num_episodes': 34,
        'episode_len': 7692,
        'camera_names': ['camera1']
    },
    'sim_vr_task_pw_5': {
        'dataset_dir': r'D:\CLOVR_Output\PW_DATA_3\HDF5',
        'num_episodes': 55,
        'episode_len': 7890,
        'camera_names': ['camera1']
    },
    'sim_vr_task_pw_6': {
        'dataset_dir': r'D:\CLOVR_Output\PW_DATA_6\HDF5',
        'num_episodes': 16,
        'episode_len': 7866,
        'camera_names': ['camera1']
    },
    'sim_vr_task_pw_7': {
        'dataset_dir': r'D:\CLOVR_Output\PW_DATA_6\HDF5',
        'num_episodes': 21,
        'episode_len': 8988,
        'camera_names': ['camera1']
    },
    'sim_vr_task_pw_8': {
        'dataset_dir': r'D:\CLOVR_Output\PW_DATA_7\HDF5',
        'num_episodes': 38,
        'episode_len': 2499,
        'camera_names': ['camera1']
    },
    'sim_vr_task_pw_9': {
        'dataset_dir': r'D:\CLOVR_Output\PW_DATA_7\HDF5',
        'num_episodes': 21,
        'episode_len': 8858,
        'camera_names': ['camera1']
    },
    'sim_vr_task_bs_fb_2': {
        'dataset_dir': r'D:\CLOVR_Output\BS_FB_DATA_1\HDF5',
        'num_episodes': 31,
        'episode_len': 7669,
        'camera_names': ['camera1']
    },
}

### Simulation envs fixed constants
DT = 0.02
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
