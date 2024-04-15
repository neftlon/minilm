#!/usr/bin/python

import glob, json, typing

class JupyterNotebook(typing.NamedTuple):
  filename: str

  def __str__(self):
    with open(self.filename) as f:
      return "\n".join(
        "".join(cell["source"])
        for cell in json.load(f)["cells"] 
        if cell["cell_type"] == "code"
      )

class PythonFile(typing.NamedTuple):
  filename: str

  def __str__(self):
    with open(self.filename) as f:
      return f.read()

def raw():
  find = lambda ext: glob.iglob("./**/*." + ext, recursive=True)
  files = []
  files += [PythonFile(name) for name in find("py")]
  files += [JupyterNotebook(name) for name in find("ipynb")]

  # join all found files together to form a dataset
  return "\n".join(str(f) for f in files)

if __name__ == "__main__":
  print(raw())
