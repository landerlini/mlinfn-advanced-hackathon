import os
from contextlib import contextmanager

@contextmanager
def working_dir(path: str):
  previous_dir = os.getcwd()
  try:
    os.chdir(path)
    yield
  finally:
    os.chdir(previous_dir)

