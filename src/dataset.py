"""Dataset handling for MOVi"""

import os

import tensorflow_datasets as tfds

from src.config import cfg
import data.movi_a
import data.movi_b


# Constants for mapping configuration to features in MOVI A and B datasets
MOVI_A = 'movi_a'
MOVI_B = 'movi_b'

# Feature type constants
INPUT = 'input'
PRIOR = 'prior'
OUTPUT = 'output'
LABEL = 'label'

# Input feature type constants
VIDEO = 'video'
DEPTH = 'depth'
OPT_FLOW = 'opt_flow'
MASK = 'mask'

# Prior feature type constants
CAM_POS = 'cam_pos'
CAM_ORI = 'cam_ori'
OBJ_CLASS = 'obj_class'

# Output feature type constants
SIZE = 'size'
POS = 'pos'
ORI = 'ori'
LIN_VEL = 'lin_vel'
ANG_VEL = 'ang_vel'
MATERIAL = 'material'
COLOR = 'color'

# Additional feature type constants
CAM = 'cam'
META = 'meta'
BACKGROUND = 'background'

# Configuration to feature mapping for MOVI A and B datasets
CFG2FEAT = {}
CFG2FEAT[MOVI_A] = {
    INPUT : {
        VIDEO : VIDEO,
        DEPTH : DEPTH,
        OPT_FLOW : 'forward_flow',
        MASK : 'segmentations'
    },
    PRIOR : {
        CAM_POS : ['camera','positions'],
        CAM_ORI : ['camera', 'quaternions'],
        OBJ_CLASS : ['instances', 'shape_label'],
    },
    OUTPUT : {
        SIZE : ['instances', 'size_label'],
        POS : ['instances', 'positions'],
        ORI : ['instances', 'quaternions'],
        LIN_VEL : ['instances', 'velocities'],
        ANG_VEL : ['instances', 'angular_velocities'],
        MATERIAL :['instances', 'material_label'],
        COLOR : ['instances', 'color_label']
    },
}
CFG2FEAT[MOVI_B] = {
    INPUT : CFG2FEAT[MOVI_A][INPUT],
    PRIOR: CFG2FEAT[MOVI_A][PRIOR],
    OUTPUT : {
        SIZE : ['instances', 'scale'],
        POS : CFG2FEAT[MOVI_A][OUTPUT][POS],
        ORI : CFG2FEAT[MOVI_A][OUTPUT][ORI],
        LIN_VEL : CFG2FEAT[MOVI_A][OUTPUT][LIN_VEL],
        ANG_VEL : CFG2FEAT[MOVI_A][OUTPUT][ANG_VEL],
        MATERIAL : CFG2FEAT[MOVI_A][OUTPUT][MATERIAL],
        COLOR : ['instances', 'color']
    }
}

# Constants for feature formatting types
RANGE = 'range'
ANGLE = 'angle'
ONE_HOT = 'one_hot'

# Configuration constant for number of classes in a feature
NUM_CLASS = 'num_class'

# Configuration for feature formatting for MOVI A and B datasets
FORMAT_FEATS = {}
FORMAT_FEATS[MOVI_A] = {
    INPUT : {
        DEPTH : RANGE,
        OPT_FLOW : RANGE
    },
    PRIOR : {
        CAM_ORI : ANGLE,
        OBJ_CLASS : ONE_HOT,
    },
    OUTPUT : {
        ORI : ANGLE,
        MATERIAL : ONE_HOT,
        COLOR : ONE_HOT
    }
}
FORMAT_FEATS[MOVI_B] = {
    INPUT : FORMAT_FEATS[MOVI_A][INPUT],
    PRIOR : FORMAT_FEATS[MOVI_A][PRIOR],
    OUTPUT : {
        ORI : FORMAT_FEATS[MOVI_A][OUTPUT][ORI],
        MATERIAL : FORMAT_FEATS[MOVI_A][OUTPUT][MATERIAL],
    }
}

# Constants for sequential features in MOVI A and B datasets
SEQ_FEATS = {}
SEQ_FEATS[MOVI_A] = {
     INPUT : [VIDEO, DEPTH, OPT_FLOW, MASK], 
     PRIOR : [CAM_POS, CAM_ORI],
     OUTPUT : [POS, ORI, LIN_VEL, ANG_VEL]
}
SEQ_FEATS[MOVI_B] = SEQ_FEATS[MOVI_A]

# Constants for features forecasted by training stage
STAGE_1 = 'stage_1'
STAGE_2 = 'stage_2'

FORECAST_FEATS = {
    STAGE_1 : [DEPTH, MASK],
    STAGE_2 : [VIDEO, MASK]
}


def load_eval_data(cfg):
    """
    Loads and returns the evaluation data for development.

    Args:
        cfg: A configuration object containing dataset.

    Returns:
        A tuple containing the training and validation datasets and dataset information.
    """
     # remove when running with full dataset
    full_train_count = 19
    split = f'train[:{full_train_count}]'

    eval_ds, ds_info = tfds.load(cfg.DATA.set, data_dir=cfg.DATA.path,
                                            split=split, 
                                            shuffle_files=False,
                                            with_info=True)
    
    eval_ds = tfds.as_numpy(eval_ds)

    return eval_ds, ds_info


def load_test_data(cfg):
    """
    Loads and returns the test dataset based on the provided configuration.

    Args:
        cfg: A configuration object containing dataset parameters.

    Returns:
        The test dataset and its information.
    """
    # remove when running with full dataset
    test_count = 8
    split = f'test[:{test_count}]'
    test_ds, ds_info = tfds.load(cfg.DATA.set, data_dir=cfg.DATA.path,
                                 split=split, 
                                 shuffle_files=False,
                                 with_info=True)

    test_ds = tfds.as_numpy(test_ds)

    return test_ds, ds_info