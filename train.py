#!/usr/bin/env python
import functools
import jax, jax.numpy as jnp, jax.random as jr, tqdm
from model import Miniformer
import checkpoint, config, optim

# get experiment configuration
hparams = config.get_config()
key = jr.key(1337)
tok = hparams.load_tokenizer()

# load dataset
text_dataset = hparams.get_text_dataset()
train, test = text_dataset.split(hparams.train_test_split_fraction)
train, test = (s.encode(tok) for s in (train, test))
del text_dataset # only the encoded versions should be used from now on

# instantiate model and optimizer
model = Miniformer.from_spec(hparams.seq_len, hparams.num_blocks, tok.num_tokens, hparams.emb_dim, hparams.num_heads, hparams.hidden_dim)
key, subkey = jr.split(key)
params = model.init(subkey)
num_params = sum(jax.tree.flatten(jax.tree.map(lambda p: p.size, params))[0])
opt = optim.Adam(alpha=hparams.learning_rate)
opt_state = opt.init(params)
print(f"num params: {num_params / 1e6:.03f}m")

@functools.partial(jax.jit, static_argnames=["model", "split", "subset_size"])
def acc(model, params, key, split, subset_size=500):
  split = {"train": train, "test": test}[split]
  batch = split.sample(key, model.seq_len + 1, num_samples=subset_size)
  x, y = batch[:,:-1], batch[:,1:]
  return jnp.mean(model(params, x).argmax(-1) == y)

def loss(model, params, x, y):
  y_pred = model(params, x)
  logits = jax.nn.log_softmax(y_pred)
  return -jnp.mean(jnp.sum(jax.nn.one_hot(y, tok.num_tokens) * logits, axis=-1))

@functools.partial(jax.jit, static_argnames=["model", "opt"])
def update_step(model, params, opt, opt_state, key):
  # generate a batch of data
  batch = train.sample(key, model.seq_len + 1, num_samples=hparams.batch_size)
  x, y = batch[:,:-1], batch[:,1:]
  # compute loss and gradients
  lossval, grads = jax.value_and_grad(loss, argnums=1)(model, params, x, y)
  # update parameters
  opt_state, params = opt.step(opt_state, params, grads)
  return lossval, opt_state, params

# run training
for epoch in (pbar := tqdm.trange(1, 1 + hparams.num_epochs)):
  # run update
  key, subkey = jr.split(key)
  lossval, opt_state, params = update_step(model, params, opt, opt_state, subkey)

  # bookkeeping via progress bar
  if (epoch % hparams.log_interval) == 0:
    stats = {"nll": lossval.item()}
    key, subkey = jr.split(key)
    for short, split in (("acc_tr", "train"), ("acc_te", "test")):
      stats[short] = acc(model, params, subkey, split).item()
    pbar.write(", ".join(f"%s=%.03f" % tup for tup in stats.items()))

# save model
if hparams.save_dir is not None:
  checkpoint.save(model, params, save_dir=hparams.save_dir)
