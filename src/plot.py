"""Helper functions for plotting."""

import matplotlib.pyplot as plt
import matplotlib as mpl

# Latex document Text width
latex_width = 412.56497

def set_mpl_format():
    #mpl.rcParams['figure.dpi'] = 500
    mpl.rcParams['font.size'] = 9
    mpl.rcParams['axes.titlesize'] = 9
    mpl.rcParams['axes.labelsize'] = 9
    mpl.rcParams['xtick.labelsize'] = 9
    mpl.rcParams['ytick.labelsize'] = 9
    mpl.rcParams['legend.fontsize'] = 9
    mpl.rcParams['legend.title_fontsize'] = 9

def set_fig_size(width=latex_width, height=latex_width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.
    
    Credit to Jack Walton for the function.
    Source: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    """

    fig_width_pt = width * fraction
    fig_height_pt = height * fraction
    
    inches_per_pt = 1 / 72.27
    
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_height_pt * inches_per_pt * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)