#!/usr/bin/python
# this code is copied from https://github.com/karpathy/minbpe/blob/master/minbpe/regex.py

import collections, dataclasses, functools, json, keyword, os, regex as re

@dataclasses.dataclass
class Bpe:
  vocab: dict[int, bytes]
  merges: dict[(int, int), int]
  special_tokens: dict[str, int]
  split_pattern: str
  compiled_split_pattern: re.Pattern

  def encode_ordinary(self, text):
    chunks = re.findall(self.compiled_split_pattern, text) if self.split_pattern is not None else [text]
    res = []
    for chunk in chunks:
      chunk = chunk.encode("utf-8")
      res.extend(self._encode_chunk(chunk))
    return res

  def encode(self, text, allowed_special="none_raise"):
    if allowed_special == "all":
      special = self.special_tokens
    elif allowed_special == "none":
      special = {}
    elif allowed_special == "none_raise":
      special = {}
      assert all(token not in text for token in self.special_tokens)
    else:
      raise ValueError("unknown mode for treating special values")
    if not special:
      return self.encode_ordinary(text)
    
    special_pat = "(" + "|".join(re.escape(k) for k in special) + ")"
    special_chunks = re.split(special_pat, text)

    res = []
    for chunk in special_chunks:
      if chunk in special:
        res.append(special[chunk])
      else:
        res.extend(self.encode_ordinary(chunk))
    return res
  
  def decode(self, enc):
    res = []
    for i in enc:
      if i in self.vocab:
        res.append(self.vocab[i])
      elif i in self.inv_special_tokens:
        res.append(self.inv_special_tokens[i].encode("utf-8"))
      else:
        raise ValueError("invalid token: %d" % i)
    res = b"".join(res)
    return res.decode("utf-8", errors="replace")

  @classmethod
  def train(cls, data, vocab_size: int, pat: str = None, verbose: bool = False, special_tokens: dict[str, int] = None):
    # initial vocab
    vocab = {i: bytes((i,)) for i in range(256)} # every byte gets an index
    assert (num_merges := vocab_size - len(vocab)) > 0

    # encode chunks and split them if a splitting pattern is provided
    if pat is not None:
      compiled_pat = re.compile(pat)
      chunks = compiled_pat.findall(data)
      chunks = [chunk.encode("utf-8") for chunk in chunks]
    else:
      compiled_pat = None
      chunks = [data.encode("utf-8")]

    # merge most common pair
    merges = {}
    for merge_idx in range(num_merges):
      # count occurrence count of each pair
      stats = collections.Counter()
      for chunk in chunks:
        stats += collections.Counter(zip(chunk, chunk[1:]))
      # find most occurring pair
      pair = max(stats, key=stats.get)
      assert pair is not None, "no more pairs"
      # keep mapping udpated
      new = len(vocab)
      a, b = pair
      vocab[new], merges[pair] = vocab[a] + vocab[b], new
      # replace occurrences of pair with new index
      chunks = [Bpe._replace_pair(chunk, pair, new) for chunk in chunks]
      if verbose:
        print(f"merge {merge_idx+1}/{num_merges}: {pair} -> {new} ({vocab[new]}) occurred {stats[pair]} times")
    
    if special_tokens is None:
      special_tokens = {}
    return cls(vocab, merges, special_tokens, pat, compiled_pat)

  @staticmethod
  def _replace_pair(data, pair, new):
    i = 0
    res = []
    while i < len(data):
      if i < len(data) - 1 and (data[i], data[i+1]) == pair:
        res.append(new)
        i += 2
      else:
        res.append(data[i])
        i += 1
    return res
  
  def _encode_chunk(self, text: bytes):
    res = list(text)
    # re-apply merges
    while len(res) >= 2: # something could be merged
      stats = collections.Counter(zip(res, res[1:]))
      pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
      if pair not in self.merges:
        break
      idx = self.merges[pair]
      res = self._replace_pair(res, pair, idx)
    return res

  @functools.cached_property
  def inv_special_tokens(self):
    return {i: t for t, i in self.special_tokens.items()}
  
  @staticmethod
  def python_pattern():
    "A splitting pattern suitable for the Python programming language"
    atoms = ":><=@#$%^.,/"
    delims = [r"\(", r"\)", r"\{", r"\}", r"\[", r"\]", "==", ">=", "<=",r"\\"] + list(atoms) + keyword.kwlist
    return r"\w+|" + "|".join(delims) + r"|\S+|\n|\s+"
  
  def dump(self, f):
    json.dump({
      "vocab": {i: list(bs) for i, bs in self.vocab.items()}, # convert bytes to list of ints
      "inv_merges": {i: tup for tup, i in self.merges.items()}, # invert merges mapping for JSON
      "special_tokens": self.special_tokens,
      "split_pattern": self.split_pattern,
    }, f)

  @classmethod
  def load(cls, f):
    d = json.load(f)
    vocab = {int(i): bytes(bs) for i, bs in d["vocab"].items()}
    merges = {tuple(tup): int(i) for i, tup in d["inv_merges"].items()}
    return cls(
      vocab=vocab,
      merges=merges,
      special_tokens=d["special_tokens"],
      split_pattern=d["split_pattern"],
      compiled_split_pattern=re.compile(d["split_pattern"]),
    )

if __name__ == "__main__":
  # train tokenizer on whole dataset
  import dataset
  tok = Bpe.train(
    dataset.raw(), 256 + 3,
    Bpe.python_pattern(),
    verbose=True,
  )

  # dump tokenizer to models directory
  models_path = "models"
  os.makedirs(models_path, exist_ok=True)
  tokenizer_path = os.path.join(models_path, "tokenizer.json")
  with open(tokenizer_path, "w") as f:
    tok.dump(f)
