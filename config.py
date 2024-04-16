import keyword, os, typing

def python_pattern():
  "A splitting pattern suitable for the Python programming language"
  atoms = ":><=@#$%^.,/"
  delims = [r"\(", r"\)", r"\{", r"\}", r"\[", r"\]", "==", ">=", "<=",r"\\"] + list(atoms) + keyword.kwlist
  return r"\w+|" + "|".join(delims) + r"|\S+|\n|\s+"

class CodeExperimentHparams(typing.NamedTuple):
  "Hyper-parameters of the model for running the self-training experiment"
  # experiment settings
  experiment_name = "code"
  seq_len: int = 50
  
  @property
  def save_dir(self):
    return os.path.join("models", self.experiment_name)

  # dataset configuration
  train_test_split_fraction: float = .9

  # tokenizer configuration
  tokenizer_dir: str = "tokenizers"
  dest_vocab_size: int = 500
  split_pattern: str = python_pattern()

  @property
  def tokenizer_json(self):
    return os.path.join(self.tokenizer_dir, f"{self.experiment_name}.json")

  # model configuration
  num_blocks: int = 5
  emb_dim: int = 50
  num_heads: int = 4
  hidden_dim: int = 100

  # optimizer preferences
  num_epochs: int = 20000
  log_interval: int = 1000
  learning_rate: float = 5e-2
  batch_size: int = 250
    

def get_config():
  return CodeExperimentHparams()