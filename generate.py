#!/usr/bin/env python

import config, checkpoint, sys, time
import jax, jax.numpy as jnp, jax.random as jr

# get command line arguments
hparams = config.get_config()
tok = hparams.load_tokenizer()

# load model and tokenizer
model, params = checkpoint.load_checkpoint(hparams.save_dir)
if model is None or params is None:
  print("could not find checkpoint. make sure to run train.py before generate.py.")
  sys.exit(-1)
tok = hparams.load_tokenizer()

@jax.jit
def pred(key, x):
  # run prediction
  logits = jax.nn.log_softmax(model(params, x))[-1]
  # sample from logits
  key, subkey = jr.split(key)
  return key, jr.categorical(subkey, logits)

# start from random vector
key = jr.key(1337)
key, subkey = jr.split(key)
raw = jr.randint(subkey, (model.seq_len,), 0, tok.num_tokens).tolist()
while True:
  # run prediction
  x = jnp.asarray(raw[-model.seq_len:])
  key, elem = pred(key, x)
  raw.append(elem.item())
  # print result and wait
  print(tok.decode(raw[-1:]), end="", flush=True)
  time.sleep(25e-3)
