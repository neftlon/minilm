#!/usr/bin/env python3

import config, checkpoint, html
import jax, jax.numpy as jnp, jax.random as jr

class Chat:
  def __init__(self):
    # load tokenizer and model
    hparams = config.ChatExperimentHparams()
    self.tok = hparams.load_tokenizer()
    self.model, self.params = checkpoint.load_checkpoint(hparams.save_dir)
    self.forward = jax.jit(self.model.__call__) # compile model to be a bit faster

    self.atok = self.tok.special_tokens["<A:> "]
    self.btok = self.tok.special_tokens["<B:> "]

    self.context = [self.atok] # start with user input
  
  def _predict_char(self, key):
    # convert context to input vector
    if len(self.context) >= self.model.seq_len:
      # truncate context if necessary
      self.context = self.context[-self.model.seq_len:]
      x = jnp.asarray(self.context)
      readloc = self.model.seq_len - 1
    else:
      pad = self.model.seq_len - len(self.context)
      x = self.context + [0 for _ in range(pad)]
      x = jnp.asarray(x)
      readloc = len(self.context) - 1
    
    # run context through model
    pred = self.forward(self.params, x)
    assert pred.ndim == 2

    # sample from prediction
    logits = pred[readloc]
    return jr.categorical(key, logits)

  def send(self, msg: str, key, max_len: int = 50):
    response = [self.btok]
    # TODO: escape prompt
    self.context += self.tok.encode(msg + "\n") + response
    for _ in range(max_len):
      key, subkey = jr.split(key)
      char = self._predict_char(subkey).item()
      self.context.append(char)
      response.append(char)
      if response[-1] == self.atok:
        response = response[:-1]
        break
    
    # split into distinct messages
    chunks = []
    for char in response:
      assert not char == self.atok, "should not find <A:> in response"
      if char == self.btok or len(chunks) == 0:
        chunks.append([])
        if char == self.btok:
          continue
      chunks[-1] = chunks[-1] + [char]
    
    # decode and escape answers
    responses = []
    for chunk in chunks:
      response = self.tok.decode(chunk).strip()
      response = html.escape(response)
      responses.append(response)
    
    return responses

chat = Chat()
key = jr.key(1337)

while True:
  # parse input
  try:
    prompt = input("> ")
  except EOFError:
    break
  if prompt == "quit":
    break

  # print answers in a loop
  key, subkey = jr.split(key)
  responses = chat.send(prompt, subkey)
  for response in responses:
    print("!", response)
