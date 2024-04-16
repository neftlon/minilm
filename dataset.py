#!/usr/bin/env python

import glob, json, regex as re, typing, tokenizer
import jax, jax.numpy as jnp, jax.random as jr

class EncDataset(typing.NamedTuple):
  data: jax.Array

  def sample(self, key, sample_len: int, num_samples: int = None):
    "Sample a batch of size `num_samples` from this dataset. Each entry has `sample_len` many tokens."
    start_shape = () if num_samples is None else (num_samples,)
    start_ix = jr.randint(key, start_shape, 0, len(self.data) - sample_len + 1)
    if num_samples is None:
      return jax.lax.dynamic_slice_in_dim(self.data, start_ix, sample_len)
    else:
      return jax.vmap(jax.lax.dynamic_slice_in_dim, (None,0,None))(self.data, start_ix, sample_len)

class TextDataset(typing.NamedTuple):
  text: str

  def split(self, ratio=.9) -> tuple["TextDataset", "TextDataset"]:
    "Split dataset based on ratio"
    assert 0 < ratio < 1
    thres = int(ratio * len(self.text))
    return TextDataset(self.text[:thres]), TextDataset(self.text[thres:])

  def encode(self, tok: tokenizer.Bpe) -> EncDataset:
    "Convert the text dataset into a token based EncDataset"
    data = jnp.asarray(tok.encode(self.text, allowed_special="all"))
    return EncDataset(data)

class CodeDataset(TextDataset):
  @classmethod
  def create(cls):
    class File(typing.NamedTuple):
      filename: str

    class JupyterNotebook(File):
      def __str__(self):
        with open(self.filename) as f:
          return "\n".join(
            "".join(cell["source"])
            for cell in json.load(f)["cells"] 
            if cell["cell_type"] == "code"
          )

    class PythonFile(File):
      def __str__(self):
        with open(self.filename) as f:
          return f.read()
    
    # collect all notebooks and Python files
    find = lambda ext: glob.iglob("./**/*." + ext, recursive=True)
    files = []
    files += [PythonFile(name) for name in find("py")]
    files += [JupyterNotebook(name) for name in find("ipynb")]

    # join all found files together to form a dataset
    return cls(text="\n".join(str(f) for f in files))

class ChatDataset(TextDataset):
  @classmethod
  def from_whatsapp(cls, chat_file: str):
    "Take a WhatsApp chat and create a dataset from it"
    with open(chat_file) as f:
      chat = f.read()

    # extract senders and messages (and remove time information)
    lines = re.split(r"\d{1,2}/\d{1,2}/\d{1,2}, \d{2}:\d{2} - ([\w+ ?]+): ", chat)[1:]
    senders, messages = lines[::2], lines[1::2]

    # replace senders with generic names
    assert len(set(senders)) == 2, "expected the chat to only have two members"
    mapping = {sender: gen for gen, sender in zip("AB", set(senders))}
    senders = [f"<{mapping[sender]}:> " for sender in senders]

    # create text
    text = ""
    for sender, message in zip(senders, messages):
      text = text + sender + message
    
    return cls(text=text)

if __name__ == "__main__":
  print(CodeDataset.create().text)
