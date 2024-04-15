import datetime, os, pickle, typing
import jax, jax.numpy as jnp

class CheckpointDef(typing.NamedTuple):
  model: typing.Any
  param_tree_def: typing.Any

def save(model, params, save_dir="models"):
  # create save directory with timestamp
  timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
  save_dir = os.path.join(save_dir, timestamp)
  os.makedirs(save_dir, exist_ok=True)
  
  # save parameters
  flat_params, tree_def = jax.tree.flatten(params)
  with open(os.path.join(save_dir, "params.npy"), "wb") as f:
    for p in flat_params:
      jnp.save(f, p)

  # save model architecture and param tree structure
  chdef = CheckpointDef(model, tree_def)
  with open(os.path.join(save_dir, "def.pk"), "wb") as f:
    pickle.dump(chdef, f)

def load_checkpoint(save_dir="models"):
  # query available models
  required_files = ("def.pk", "params.npy")
  names = os.listdir(save_dir)
  names = (
    name for name in names
    # check if all required files exist
    if all(os.path.exists(os.path.join(save_dir, name, f)) for f in required_files)
  )
  names = sorted(names, reverse=True)

  # if found, return the most recent model
  if len(names) > 0:
    name = names[0] # the most recent directory by timestamp

    # load architecture and param tree structure
    with open(os.path.join(save_dir, name, "def.pk"), "rb") as f:
      chdef = pickle.load(f)

    # load params
    flat_params = []
    with open(os.path.join(save_dir, name, "params.npy"), "rb") as f:
      for _ in range(chdef.param_tree_def.num_leaves):
        flat_params.append(jnp.load(f))
    params = jax.tree.unflatten(chdef.param_tree_def, flat_params)
    
    return chdef.model, params
  return None, None # no checkpoint found