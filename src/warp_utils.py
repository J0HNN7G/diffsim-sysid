"""Render physics simulation to USD using Warp's rendering API."""

from tqdm import tqdm
import numpy as np
import warp.sim.render as wpr

epsilon = 1e-6

def render_usd(stage, model, states, sim_duration, sim_dt):
    """
    Render a physics simulation to USD using Warp's rendering API.

    Args:
        stage (str): The path to the USD stage file.
        model (warp.sim.Model): Physics simulation model
        states (list): List of simulation states to render.
        sim_duration (float): Duration of the simulation in seconds.
        sim_dt (float): Time step of the simulation in seconds.

    Renders the provided physics simulation states to USD using Warp's rendering API.
    The rendered frames are saved to a USD file specified by `model`.

    Example:
        # Render physics simulation to USD
        model_path = 'path/to/model.usd'
        states = [...]  # List of simulation states
        sim_duration = 10.0  # Duration of the simulation (in seconds)
        sim_dt = 0.1  # Time step of the simulation (in seconds)
        usd_render(model_path, states, sim_duration, sim_dt)
    """
    renderer = wpr.SimRenderer(model, stage, scaling=1.0, up_axis='Z')
    
    sim_time = np.arange(0, sim_duration + epsilon, sim_dt)
    for i in tqdm(range(len(sim_time))):
        renderer.begin_frame(sim_time[i])
        renderer.render(states[i])
        renderer.end_frame()

    renderer.save()
