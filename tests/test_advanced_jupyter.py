import os
from working_dir import working_dir
from snakemake import snakemake

def test_snakemake():
  with working_dir("advanced_jupyter"):
    ret = snakemake("Snakefile", forceall=True, targets=['collect'], cores=8)
    if not ret:
      raise RuntimeError("Something went wrong executing snakemake")

