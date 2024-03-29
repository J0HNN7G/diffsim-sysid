"""Dataset handling for MOVi"""
import tensorflow_datasets as tfds
import src.data.movi_a


def load_train_data(cfg):
    """
    Loads and returns the training data for development.

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


def load_eval_data(cfg):
    """
    Loads and returns the eval dataset based on the provided configuration.

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