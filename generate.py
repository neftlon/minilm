#!/usr/bin/env python

import argparse, checkpoint, sys, time, tokenizer
import jax, jax.numpy as jnp, jax.random as jr

# get command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--tokenizer", type=argparse.FileType("r"), default="models/tokenizer.json")
args = parser.parse_args()

# load model and tokenizer
model, params = checkpoint.load_checkpoint()
if model is None or params is None:
  print("could not find checkpoint. make sure to run train.py before generate.py.")
  sys.exit(-1)
tok = tokenizer.Bpe.load(args.tokenizer)
vocab_size = len(tok.vocab)

@jax.jit
def pred(key, x):
  # run prediction
  logits = jax.nn.log_softmax(model(params, x))[-1]
  # sample from logits
  key, subkey = jr.split(key)
  return key, jr.categorical(subkey, logits)

# start from random vector
L = 25 # TODO: where can this be fetched?
key = jr.key(1337)
key, subkey = jr.split(key)
raw = jr.randint(subkey, (L,), 0, vocab_size).tolist()
while True:
  # run prediction
  x = jnp.asarray(raw[-L:])
  key, elem = pred(key, x)
  raw.append(elem.item())
  # print result and wait
  print(tok.decode(raw[-1:]), end="", flush=True)
  time.sleep(50e-3)
