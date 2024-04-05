"""Differentiable Simulator"""

import os
from copy import deepcopy
# math
import numpy as np
import torch
import pytorch3d
import pytorch3d.transforms
# physics
import openmesh
import warp as wp
import warp.sim as wps
from src.warp_utils import render_usd
# rendering
from mitsuba.scalar_rgb import Transform4f as mit


#######
# Sim #
#######


# simulation parameters
SIM_DURATION = 2.0
# control frequency
FPS = 12
FRAME_DT = 1.0 / float(FPS)
FRAME_STEPS = int(SIM_DURATION / FRAME_DT)
# sim frequency
SIM_FPS = 240
SIM_SUBSTEPS = int(SIM_FPS / FPS)
SIM_STEPS = FRAME_STEPS * SIM_SUBSTEPS
SIM_DT = FRAME_DT / SIM_SUBSTEPS

# world parameters
UP = (0, 0, 1)
GRAVITY = -10.0

# contact parameters
KE = 500.0
KD = 25.0
KF = 50.0


##########
# Render #
##########

RENDER_ORIG = np.array([0, 0, 0])
RENDER_UP = np.array([0, 0, 1])

# integrator
RENDER_INTEGRATOR = 'path'
RENDER_HIDE_EMITTERS = True

# camera
RENDER_CAM_TYPE = 'perspective'

RAD2DEG = 180 / np.pi

RENDER_CAM_FILM = 'hdrfilm'
RENDER_CAM_RFILTER_TYPE = 'gaussian'
RENDER_CAM_RFILTER_STTDEV = 0.5
RENDER_CAM_SAMPLE_BORDER = True

# ground
GROUND_TYPE = 'rectangle'

GROUND_SCALE = 1000.0
GROUND_TO_WORLD = mit.scale(GROUND_SCALE)

GROUND_BSDF_TYPE = 'principled'
GROUND_BASE_COLOR_TYPE = 'rgb' 
GROUND_RGB = [0.5, 0.5, 0.5]
GROUND_SPEC = 0.0
GROUND_ROUGH = 1.0


# lights
LAMP_SCALE = 0.5

SUN_LIGHT_TYPE = 'directional'
SUN_LIGHT_POS = [11.6608, -6.62799, 25.8232]
SUN_LIGHT_TO_WORLD = mit.look_at(SUN_LIGHT_POS, RENDER_ORIG, RENDER_UP)
SUN_LIGHT_INTENSITY = 0.45
SUN_LIGHT_IRRADIANCE_TYPE = 'rgb'

LAMP_BACK_TYPE = 'point'
LAMP_BACK_POS = [-1.1685, 2.64602, 5.81574]
LAMP_BACK_TO_WORLD = mit.look_at(LAMP_BACK_POS, RENDER_ORIG, RENDER_UP).scale(LAMP_SCALE)
LAMP_BACK_INTENSITY_TYPE = 'rgb'
LAMP_BACK_INTENSITY = 50.0 / 3

LAMP_KEY_TYPE = 'point'
LAMP_KEY_POS = np.array([6.44671, -2.90517, 4.2584])
LAMP_KEY_TO_WORLD = mit.look_at(LAMP_KEY_POS, RENDER_ORIG, RENDER_UP).scale(LAMP_SCALE)
LAMP_KEY_INTENSITY_TYPE = 'rgb'
LAMP_KEY_RGB = np.array([255/255.0, 237/255.0, 208/255.0])
LAMP_KEY_INTENSITY = 100.0 / 3
LAMP_KEY_RADIANCE = LAMP_KEY_INTENSITY * LAMP_KEY_RGB

LAMP_FILL_TYPE = 'point'
LAMP_FILL_POS = [-4.67112, -4.0136, 3.01122]
LAMP_FILL_TO_WORLD = mit.look_at(LAMP_FILL_POS, RENDER_ORIG, RENDER_UP).scale(LAMP_SCALE)
LAMP_FILL_RGB = np.array([226/255.0, 233/255.0, 255/255.0])
LAMP_FILL_INTENSITY_TYPE = 'rgb'
LAMP_FILL_INTENSITY = 30.0 / 3
LAMP_FILL_RADIANCE = LAMP_FILL_INTENSITY * LAMP_FILL_RGB


ENV_LIGHT_TYPE = 'constant'
ENV_LIGHT_INTENSITY = 0.05
ENV_LIGHT_RADIANCE_TYPE = 'rgb'


# objects
CLEVR_SHAPE_NAMES = ("cube", "cylinder", "sphere")
CLEVR_SIZE_NAMES = ('small', 'large')
CLEVR_SIZES = (0.7, 1.4)
CLEVR_COLOR_NAMES = ("blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow")
CLEVR_COLORS = ([42/255.0, 75/255.0, 215/255.0], 
                [129/255.0, 74/255.0, 25/255.0], 
                [41/255.0, 208/255.0, 208/255.0], 
                [87/255.0, 87/255.0, 87/255.0], 
                [29/255.0, 105/255.0, 20/255.0], 
                [129/255.0, 38/255.0, 192/255.0], 
                [173/255.0, 35/255.0, 35/255.0], 
                [255/255.0, 238/255.0, 5/255.0])
CLEVR_MATERIAL_NAMES = ("metal", "rubber")

# bsdf
RUBBER_BSDF_DICT = {
    'type': 'principled',
    'base_color': {
        'type': 'rgb'
    },
    'metallic' : 0.0,
    'roughness' : 0.7,
    'eta' : 1.25,
    #'specular' : 0.33
}
METAL_BSDF_DICT = {
    'type': 'principled',
    'base_color': {
        'type': 'rgb'
    },
    'metallic' : 1.0,
    'roughness' : 0.2,
    'eta' : 2.5
}



###########
# Physics #
###########

# ground parameters
CLEVR_GROUND_MU = 0.3
CLEVR_GROUND_RESTITUTION = 0.5

# object parameters
CLEVR_DENSITIES = (2.7, 1.1)

CLEVR_SHAPE_OBJ_FPS = {
    'cube': os.path.join('assets', 'cube', 'collision_geometry.obj'),
    'cylinder': os.path.join('assets', 'cylinder', 'collision_geometry.obj'),
    'sphere': os.path.join('assets', 'sphere', 'collision_geometry.obj')
}

CLEVR_SHAPE_BASE_MASS = {
    'cube': 0.9936,
    'cylinder': 0.7808,
    'sphere': 0.5225
}

CLEVR_SHAPE_BASE_INERTIA = {
    'cube': [
        [0.1643, 0.0, 0.0],
        [0.0, 0.1643, 0.0],
        [0.0, 0.0, 0.1643]
    ],
    'cylinder': [
        [0.1131, 0.0, 0.0],
        [0.0, 0.1131, 0.0],
        [0.0, 0.0, 0.0971]
    ],
    'sphere': [
        [0.0522, 0.0, 0.0],
        [0.0, 0.0522, 0.0],
        [0.0, 0.0, 0.0522]
    ]
}


class Simulator():
    def __init__(self, cfg, sample):
        self.cfg = cfg
        self.sample = sample 
        self.setup_physics_scene()
        self.setup_render_scene()


    def set_sample(self, sample):
        self.clear()
        self.sample = sample
        self.setup_physics_scene()
        self.setup_render_scene()


    def setup_render_scene(self):
        self.render_dict = {
            'type' : 'scene'
        }

        # integrator
        integrator_dict = {
            'type': RENDER_INTEGRATOR,
            'hide_emitters' : RENDER_HIDE_EMITTERS
        }
        self.render_dict['integrator'] = integrator_dict

        # camera
        cam_width = int(self.sample['metadata']['width'])
        cam_height = int(self.sample['metadata']['height'])

        cam_pos = self.sample['camera']['positions'][0]
        cam_to_world = mit.look_at(cam_pos, RENDER_ORIG, RENDER_UP)

        cam_fov = self.sample['camera']['field_of_view'] * RAD2DEG

        cam_dict = {
            'type': RENDER_CAM_TYPE,
            'to_world': cam_to_world,
            'fov': cam_fov,
            'film': {
                'type': RENDER_CAM_FILM,
                'width': cam_width,
                'height': cam_height,
                'rfilter': { 
                    'type': RENDER_CAM_RFILTER_TYPE,
                    'stddev': RENDER_CAM_RFILTER_STTDEV
                    },
                'sample_border': RENDER_CAM_SAMPLE_BORDER
            },
        }
        self.render_dict['cam'] = cam_dict

        # ground
        ground_dict = {
            'type': GROUND_TYPE,
            'to_world': GROUND_TO_WORLD,
            'bsdf': {
                'type': GROUND_BSDF_TYPE,
                'base_color': {
                    'type': GROUND_BASE_COLOR_TYPE,
                    'value': GROUND_RGB
                },
                'specular': GROUND_SPEC,
                'roughness': GROUND_ROUGH
            }
        }
        self.render_dict['ground'] = ground_dict

        # lights
        sun_light_dict = {
                'type': SUN_LIGHT_TYPE,
                'to_world': SUN_LIGHT_TO_WORLD,
                'irradiance': {
                    'type': SUN_LIGHT_IRRADIANCE_TYPE,
                    'value': SUN_LIGHT_INTENSITY
                }
        }
        self.render_dict['sun'] = sun_light_dict

        lamp_back_dict = {
                'type': LAMP_BACK_TYPE,
                'to_world': LAMP_BACK_TO_WORLD,
                'intensity': {
                    'type': LAMP_BACK_INTENSITY_TYPE,
                    'value': LAMP_BACK_INTENSITY
                }
        }
        self.render_dict['lamp_back'] = lamp_back_dict

        lamp_key_dict = {
                'type': LAMP_KEY_TYPE,
                'to_world': LAMP_KEY_TO_WORLD,
                'intensity': {
                    'type': LAMP_KEY_INTENSITY_TYPE,
                    'value': LAMP_KEY_RADIANCE
                }
        }
        self.render_dict['lamp_key'] = lamp_key_dict

        lamp_fill_dict = {
                'type': LAMP_FILL_TYPE,
                'to_world': LAMP_FILL_TO_WORLD,
                'intensity': {
                    'type': LAMP_FILL_INTENSITY_TYPE,
                    'value': LAMP_FILL_RADIANCE
                    }
        }
        self.render_dict['lamp_fill'] = lamp_fill_dict

        env_light_dict = {
            'type': ENV_LIGHT_TYPE,
            'radiance': {
                'type': ENV_LIGHT_RADIANCE_TYPE,
                'value': ENV_LIGHT_INTENSITY
            }
        }
        self.render_dict['env_light'] = env_light_dict


    def get_render_orig_timestep_dict(self, time_step):
        timestep_dict = dict(self.render_dict)
        # add all objects
        for obj_idx in range(self.sample['metadata']['num_instances']):
            obj_class_idx = self.sample['instances']['shape_label'][obj_idx]
            obj_class = CLEVR_SHAPE_NAMES[obj_class_idx]

            obj_scale_idx = self.sample['instances']['size_label'][obj_idx]
            obj_scale = CLEVR_SIZES[obj_scale_idx] / 2.0

            obj_pos = self.sample['instances']['positions'][obj_idx][time_step]
            obj_rot = self.sample['instances']['quaternions'][obj_idx][time_step]
            obj_aa = pytorch3d.transforms.quaternion_to_axis_angle(torch.tensor(obj_rot)).detach().numpy()
            obj_aa_norm = np.linalg.norm(obj_aa)
            obj_angle = obj_aa_norm * RAD2DEG
            obj_axis = obj_aa / obj_aa_norm 
            if obj_class == CLEVR_SHAPE_NAMES[1]:
                obj_scale = [obj_scale, obj_scale, 2 * obj_scale]
            obj_to_world = mit.translate(obj_pos).rotate(axis=obj_axis, angle=obj_angle).scale(obj_scale)

            obj_rgb_idx = self.sample['instances']['color_label'][obj_idx]
            obj_rgb = CLEVR_COLORS[obj_rgb_idx]

            obj_material_idx = self.sample['instances']['material_label'][obj_idx]

            if obj_material_idx == 0:
                bsdf_dict = deepcopy(METAL_BSDF_DICT)
                bsdf_dict['base_color']['value'] = obj_rgb
            elif obj_material_idx == 1:
                bsdf_dict = deepcopy(RUBBER_BSDF_DICT)
                bsdf_dict['base_color']['value'] = obj_rgb

            if obj_class_idx == 1:   
                # cylinders are made of several shapes
                shape_group = {
                    'type' : 'shapegroup',
                    'body' : {
                        'type' : 'cylinder',
                        'p0' : np.array([0, 0, -0.5]),
                        'p1' : np.array([0, 0, 0.5]),
                        'bsdf' : bsdf_dict
                    },
                    'end1' : {
                        'type' : 'disk',
                        'to_world' : mit.translate([0, 0, -0.5]),
                        'flip_normals' : True,
                        'bsdf' : bsdf_dict
                    },
                    'end2' : {
                        'type' : 'disk',
                        'to_world' : mit.translate([0, 0, 0.5]),
                        'bsdf' : bsdf_dict
                    }
                }
                obj_dict = {
                    'type': 'instance',
                    'to_world': obj_to_world,
                    'shapegroup': shape_group
                }
            else:
                obj_dict = {
                        'type': obj_class,
                        'to_world': obj_to_world,
                        'bsdf' : bsdf_dict
                    }
            timestep_dict[f'obj_{obj_idx}'] = obj_dict

        return timestep_dict
    

    def get_render_timestep_dict(self, time_step):
        timestep_dict = dict(self.render_dict)
        # add all objects
        for obj_idx in range(self.sample['metadata']['num_instances']):
            obj_class_idx = self.sample['instances']['shape_label'][obj_idx]
            obj_class = CLEVR_SHAPE_NAMES[obj_class_idx]

            obj_scale_idx = self.sample['instances']['size_label'][obj_idx]
            obj_scale = CLEVR_SIZES[obj_scale_idx] / 2.0

            obj_pos = self.phys_states[time_step].body_q.numpy()[obj_idx][:3]
            obj_rot = self.phys_states[time_step].body_q.numpy()[obj_idx][3:]
            # i,j,k,w -> i,j,k,w
            obj_rot = np.concatenate([obj_rot[3:], obj_rot[:3]])
            obj_aa = pytorch3d.transforms.quaternion_to_axis_angle(torch.tensor(obj_rot)).detach().numpy()
            obj_aa_norm = np.linalg.norm(obj_aa)
            obj_angle = obj_aa_norm * RAD2DEG
            obj_axis = obj_aa / obj_aa_norm 
            if obj_class == CLEVR_SHAPE_NAMES[1]:
                obj_scale = [obj_scale, obj_scale, 2 * obj_scale]
            obj_to_world = mit.translate(obj_pos).rotate(axis=obj_axis, angle=obj_angle).scale(obj_scale)

            obj_rgb_idx = self.sample['instances']['color_label'][obj_idx]
            obj_rgb = CLEVR_COLORS[obj_rgb_idx]

            obj_material_idx = self.sample['instances']['material_label'][obj_idx]

            if obj_material_idx == 0:
                bsdf_dict = deepcopy(METAL_BSDF_DICT)
                bsdf_dict['base_color']['value'] = obj_rgb
            elif obj_material_idx == 1:
                bsdf_dict = deepcopy(RUBBER_BSDF_DICT)
                bsdf_dict['base_color']['value'] = obj_rgb

            if obj_class_idx == 1:   
                # cylinders are made of several shapes
                shape_group = {
                    'type' : 'shapegroup',
                    'body' : {
                        'type' : 'cylinder',
                        'p0' : np.array([0, 0, -0.5]),
                        'p1' : np.array([0, 0, 0.5]),
                        'bsdf' : bsdf_dict
                    },
                    'end1' : {
                        'type' : 'disk',
                        'to_world' : mit.translate([0, 0, -0.5]),
                        'flip_normals' : True,
                        'bsdf' : bsdf_dict
                    },
                    'end2' : {
                        'type' : 'disk',
                        'to_world' : mit.translate([0, 0, 0.5]),
                        'bsdf' : bsdf_dict
                    }
                }
                obj_dict = {
                    'type': 'instance',
                    'to_world': obj_to_world,
                    'shapegroup': shape_group
                }
            else:
                obj_dict = {
                        'type': obj_class,
                        'to_world': obj_to_world,
                        'bsdf' : bsdf_dict
                    }
            timestep_dict[f'obj_{obj_idx}'] = obj_dict

        return timestep_dict
    


    def setup_physics_scene(self):
            builder = wps.ModelBuilder(up_vector=wp.vec3(UP), gravity=GRAVITY)

            # ground
            builder.set_ground_plane(ke=KE, 
                                     kd=KD, 
                                     kf=KF,
                                     mu=CLEVR_GROUND_MU, 
                                     restitution=CLEVR_GROUND_RESTITUTION)

            for obj_idx in range(self.sample['metadata']['num_instances']):
                obj_class_idx = self.sample['instances']['shape_label'][obj_idx]
                obj_class = CLEVR_SHAPE_NAMES[obj_class_idx]

                obj_mesh_fp = os.path.join(self.cfg.path, CLEVR_SHAPE_OBJ_FPS[obj_class])
                obj_base_mass = CLEVR_SHAPE_BASE_MASS[obj_class]
                obj_base_inertia = CLEVR_SHAPE_BASE_INERTIA[obj_class]

                obj_scale_idx = self.sample['instances']['size_label'][obj_idx]
                obj_scale = [CLEVR_SIZES[obj_scale_idx]] * 3

                obj_pos = self.sample['instances']['positions'][obj_idx][0]
                obj_rot = self.sample['instances']['quaternions'][obj_idx][0]
                # w,i,j,k -> i,j,k,w
                obj_rot = np.concatenate([obj_rot[1:], obj_rot[:1]])

                obj_lin_vel = self.sample['instances']['velocities'][obj_idx][0]
                obj_ang_vel = self.sample['instances']['angular_velocities'][obj_idx][0]

                obj_fric = self.sample['instances']['friction'][obj_idx]
                obj_restitution = self.sample['instances']['restitution'][obj_idx]

                obj_material = self.sample['instances']['material_label'][obj_idx]
                obj_density = CLEVR_DENSITIES[obj_material]

                # read mesh
                m = openmesh.read_trimesh(obj_mesh_fp)
                mesh_points = np.array(m.points())
                mesh_indices = np.array(m.face_vertex_indices(), dtype=np.int32).flatten()
                mesh = wps.Mesh(vertices=mesh_points, 
                                indices=mesh_indices,
                                compute_inertia=False,
                                is_solid=True)
                mesh.has_inertia = True
                mesh.I = wp.mat33(obj_base_inertia)
                mesh.mass = obj_base_mass

                # add body
                obj_body_q = wp.transform(obj_pos, obj_rot)
                obj_body_id = builder.add_body(origin=obj_body_q, name = f'obj_{obj_idx}')

                # add mesh to body
                builder.add_shape_mesh(body=obj_body_id,
                                            mesh=mesh,
                                            scale=obj_scale,
                                            density=obj_density,
                                            mu=obj_fric,
                                            ke=KE,
                                            kd=KD,
                                            kf=KF,
                                            restitution=obj_restitution,
                                            is_solid=True,
                                            has_ground_collision=True)
                
                obj_body_qd = np.concatenate([obj_ang_vel, obj_lin_vel]).tolist()
                builder.body_qd[obj_body_id] = obj_body_qd

            # model and states
            device = wp.get_cuda_device()
            model = builder.finalize(device, requires_grad=False)
            states = [model.state(requires_grad=False) for _ in range(SIM_STEPS+1)]

            self.phys_device = device
            self.phys_model = model
            self.phys_states = states
            self.phys_integrator = wp.sim.SemiImplicitIntegrator()
            self.run_phys()


    def run_phys(self):       
        for i in range(SIM_STEPS):
            self.phys_states[i].clear_forces()
            wp.sim.collide(self.phys_model, self.phys_states[i])
            self.phys_integrator.simulate(self.phys_model, self.phys_states[i], self.phys_states[i + 1], SIM_DT)


    def clear(self):
        del self.sample
        self.sample = None
        # render
        del self.render_dict
        del self.render_timestep_dict
        self.render_dict = None
        self.render_timestep_dict = None
        # physics
        del self.phys_device
        del self.phys_model
        del self.phys_states
        del self.phys_integrator

        self.phys_device = None
        self.phys_model = None
        self.phys_states = None
        self.phys_integrator = None


    def usd(self, stage):
        render_usd(stage, self.phys_model, self.phys_states, SIM_DURATION, SIM_DT)

    
