"""Config template for system identification."""
from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
cfg = CN()


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
cfg.DATA = CN()

# path to the root of the dataset directory
cfg.DATA.path = ""
# dataset name
cfg.DATA.set = ""
# max number of objects in a video
cfg.DATA.max_objs = 10
# video fps
cfg.DATA.fps = 12
# video size
cfg.DATA.fpv = 24
# video height
cfg.DATA.height = 256
# video width
cfg.DATA.width = 256


# -----------------------------------------------------------------------------
# System Identification
# -----------------------------------------------------------------------------
cfg.SYS_ID = CN()

# number of iterations
cfg.SYS_ID.iters = 4

cfg.SYS_ID.VIS = CN()
# include visual loss
cfg.SYS_ID.VIS.include = True
# number of samples per pixel for rendering
cfg.SYS_ID.VIS.spp = 4
# visual loss weight
cfg.SYS_ID.VIS.weight = 1.0

cfg.SYS_ID.GEOM = CN()
# include geometry loss
cfg.SYS_ID.GEOM.include = True
# number of samples per mesh for point clouds
cfg.SYS_ID.GEOM.spm = 100

cfg.SYS_ID.OPTIM = CN()
# algorithm to use for optimization
cfg.SYS_ID.OPTIM.optim = "adam"
# initial learning rate
cfg.SYS_ID.OPTIM.lr = 0.01
# beta1 for Adam optimizer
cfg.SYS_ID.OPTIM.beta1 = 0.9
# beta2 for Adam optimizer
cfg.SYS_ID.OPTIM.beta2 = 0.999
# L2 penalty (regularization term) parameter
cfg.SYS_ID.OPTIM.decay = 0.0


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
cfg.EVAL = CN()

cfg.EVAL.OUTPUT = CN()
# directory to evaluation outputs
cfg.EVAL.OUTPUT.path = ""

cfg.EVAL.OUTPUT.FN = CN()
# config filename
cfg.EVAL.OUTPUT.FN.config = "config.yaml"
# log filename
cfg.EVAL.OUTPUT.FN.log = "log.txt"
# estimation results
cfg.EVAL.OUTPUT.FN.pred = "pred.csv"

# parameters estimated
cfg.EVAL.PARAM = CN()

# density
cfg.EVAL.PARAM.DENSITY  = CN()
cfg.EVAL.PARAM.DENSITY.include = True
cfg.EVAL.PARAM.DENSITY.prior = 1.35

# friction
cfg.EVAL.PARAM.FRICTION  = CN()
cfg.EVAL.PARAM.FRICTION.include = True
cfg.EVAL.PARAM.FRICTION.prior = 0.5

# restitution
cfg.EVAL.PARAM.RESTITUTION  = CN()
cfg.EVAL.PARAM.RESTITUTION.include = True
cfg.EVAL.PARAM.RESTITUTION.prior = 0.5