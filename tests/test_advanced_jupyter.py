import os
from snakemake import snakemake

def test_snakemake():
  os.chdir("advanced_jupyter")
  ret = snakemake("Snakefile", forceall=True, targets=['collect'], cores=8)
  if not ret:
    raise RuntimeError("Something went wrong executing snakemake")

