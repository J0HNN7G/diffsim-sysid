"""System identification module and helper functions."""

import torch
import numpy as np

# physics
import warp as wp

from src.sim import UP, GRAVITY, CLEVR_SHAPE_NAMES, CLEVR_SIZES, CLEVR_DENSITIES, CLEVR_SHAPE_BASE_MASS

from src.warp_utils import render_usd

DEF_TIME_STEP = 12




class SysId:
    """
    System identification module.

    This class provides functionality for system identification, including setting samples and priors for the system.
    """

    def __init__(self, cfg):
        """
        Initialize system identification module.

        Parameters:
        - cfg (dict): Configuration parameters for system identification.
        """
        self.cfg = cfg
        self.priors = None

    def set_sample(self, sample, heuristic_time_step=False):
        """
        Set samples for system identification.

        Parameters:
        - sample (dict): The sample data for system identification.
        - heuristic_time_step (bool): If True, determines time step using a heuristic method; otherwise, uses default time step.

        """
        self.sample = sample
        self.time_step = DEF_TIME_STEP
        if heuristic_time_step:
            self.time_step = -1
            max_visibility = 0
            max_visibility_time_step = 1
            for i in range(1, self.cfg.DATA.fpv):
                visibility_count = np.count_nonzero(self.sample['instances']['visibility'][:, i])
                if visibility_count == self.sample['metadata']['num_instances']:
                    self.time_step = i
                    break
                else:
                    if visibility_count > max_visibility:
                        max_visibility = visibility_count
                        max_visibility_time_step = i
            if self.time_step == -1:
                self.time_step = max_visibility_time_step
    
    def set_priors(self):
        """
        Set priors for system identification.
        
        This method calculates priors for each object instance based on its physical properties.
        """
        self.inv_mass_min = torch.zeros(self.sample['metadata']['num_instances'])
        self.inv_mass_max = torch.zeros(self.sample['metadata']['num_instances'])
        self.inv_mass_prior = torch.zeros(self.sample['metadata']['num_instances'])
        self.density_prior = torch.zeros(self.sample['metadata']['num_instances'])

        # for all objects
        for obj_idx in range(self.sample['metadata']['num_instances']):
            obj_class_idx = self.sample['instances']['shape_label'][obj_idx]
            obj_class = CLEVR_SHAPE_NAMES[obj_class_idx]

            obj_scale_idx = self.sample['instances']['size_label'][obj_idx]
            obj_scale = CLEVR_SIZES[obj_scale_idx]

            obj_base_mass = CLEVR_SHAPE_BASE_MASS[obj_class]

            obj_min_mass = obj_base_mass * CLEVR_DENSITIES[1] * (obj_scale)**3
            obj_max_mass = obj_base_mass * CLEVR_DENSITIES[0] * (obj_scale)**3
            
            self.inv_mass_min[obj_idx] = 1.0 / obj_max_mass
            self.inv_mass_max[obj_idx] = 1.0 / obj_min_mass

            mean_density = (CLEVR_DENSITIES[0] + CLEVR_DENSITIES[1]) / 2.0
            self.density_prior[obj_idx] = mean_density
            self.inv_mass_prior[obj_idx] = 1.0 / (obj_base_mass * mean_density * (obj_scale)**3)