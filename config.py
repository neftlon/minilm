import argparse, dataset, keyword, os, tokenizer, typing

def python_pattern():
  "A splitting pattern suitable for the Python programming language"
  atoms = ":><=@#$%^.,/"
  delims = [r"\(", r"\)", r"\{", r"\}", r"\[", r"\]", "==", ">=", "<=",r"\\"] + list(atoms) + keyword.kwlist
  return r"\w+|" + "|".join(delims) + r"|\S+|\n|\s+"

class CodeExperimentHparams(typing.NamedTuple):
  "Hyper-parameters of the model for running the self-training experiment"
  # experiment settings
  experiment_name: str = "code"
  seq_len: int = 50

  @property
  def save_dir(self):
    return os.path.join("models", self.experiment_name)

  # dataset configuration
  train_test_split_fraction: float = .9

  def get_text_dataset(self):
    return dataset.CodeDataset.create()

  # tokenizer configuration
  tokenizer_dir: str = "tokenizers"
  dest_vocab_size: int = 500
  split_pattern: str = python_pattern()
  special_tokens: dict[str, int] = {}

  @property
  def tokenizer_json(self):
    return os.path.join(self.tokenizer_dir, f"{self.experiment_name}.json")
  
  def load_tokenizer(self):
    return tokenizer.Bpe.load(self.tokenizer_json)

  # model configuration
  num_blocks: int = 4
  emb_dim: int = 100
  num_heads: int = 4
  hidden_dim: int = 200

  # optimizer preferences
  num_epochs: int = 20000
  log_interval: int = 100
  learning_rate: float = 1e-4
  batch_size: int = 100

def chat_pattern():
  "A splitting pattern suitable for WhatsApp chats"
  return r"\:\s+|\.+|, |!+|\?\s+| \w+|\S+|\s+"

class ChatExperimentHparams(typing.NamedTuple):
  # experiment settings
  experiment_name: str = "chat"
  seq_len: int = 100

  @property
  def save_dir(self):
    return os.path.join("models", self.experiment_name)

  # dataset configuration
  train_test_split_fraction: float = .9
  chat_txt: str = "data/chat.txt"

  def get_text_dataset(self):
    return dataset.ChatDataset.from_whatsapp(self.chat_txt)
  
  # tokenizer configuration
  tokenizer_dir: str = "tokenizers"
  dest_vocab_size: int = 500
  split_pattern: str = chat_pattern()
  special_tokens: dict[str, int] = {"<A:> ": 500, "<B:> ": 501}

  @property
  def tokenizer_json(self):
    return os.path.join(self.tokenizer_dir, f"{self.experiment_name}.json")
  
  def load_tokenizer(self):
    return tokenizer.Bpe.load(self.tokenizer_json)

  # model configuration
  num_blocks: int = 6
  emb_dim: int = 200
  num_heads: int = 8
  hidden_dim: int = 1000

  # optimizer preferences
  num_epochs: int = 50000
  log_interval: int = 100
  learning_rate: float = 1e-4
  batch_size: int = 100

def get_config():
  return ChatExperimentHparams()
