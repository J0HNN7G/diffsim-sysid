"""Differentiable Simulator"""


class Simulator():
    def __init__(self, physics_dict, graphics_dict, param_list):
        self.physics_dict = physics_dict
        self.graphics_dict = graphics_dict
        self.param_list = param_list
    
    def forward(self, x):
        return self.physics_dict(x, self.param_list)