import json
import tensorflow as tf
from args import args

# ==========================================
# Initialize an empty class
# ==========================================
class ModelConfig:
    pass

# ==========================================
# Create an instance of the empty class
# ==========================================
model_config = ModelConfig()

# ==========================================
# In this instace, add all attributes from the configuration set in the .json file specified in args.model
# The name of the json file containing all the parameters is to be set in "args.model"
# ==========================================
with open(args.model, 'r') as f:
    model_dict = json.load(f)
    for k,v in model_dict.items():
        setattr(model_config, k, v)

# ==========================================
# In the .json file, all attributes are set as strings
# For some attributes, this does not make sense
# For instance, model_handle and optimizer handle are functions.
# This function 'casts' the strings appropriately.
# ==========================================
def rec_getattr(obj, name):
    names = name.split('.')
    if isinstance(obj, dict):
        ret = obj[names[0]]
    else:
        ret = getattr(obj, names[0])
    for k in names[1:]:
        ret = getattr(ret, k)
    return ret

# ==========================================
# call the function defined above the attributes that need it
# ==========================================
model_config.optimizer_handle = rec_getattr(locals(), model_config.optimizer_handle)