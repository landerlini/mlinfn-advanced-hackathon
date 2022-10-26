import os
from contextlib import contextmanager

def working_dir(path: str):
  previous_dir = os.getcwd()
  try:
    os.chdir(path)
  finally:
    os.chdir(previous_dir)

