#!/bin/env python3
# Developed for the hackathon in Padova, 2024 
# ReCaS users' volume must be mounted in /recas-data, for example via sshfs:
#   sudo apt-get install sshfs
#   sshfs recas:/data/user-data /recas-data
#

import os
from glob import glob
import re

from argparse import ArgumentParser
from pathlib import Path
import shutil

REPO=Path("/home/ubuntu/mlinfn-advanced-hackathon/")
USER_DATA_PATHS = [
            Path("/user-data/nfs"),
            Path("/recas-data/nfs"),
        ]

exercises = dict(
        lhcf=Path(REPO/"ex/lhcf-cnn/ex-student"),
        gan=Path(REPO/"ex/gan-detector/ex-student"),
        asd=Path(REPO/"ex/asd-diagnosis/ex-student"),
        qml=Path(REPO/"ex/quantum-ml/ex-student"),
        infra=Path(REPO/"infrastructure_handson.ipynb"),
        )


solutions = dict(
        lhcf=Path(REPO/"ex/lhcf-cnn/ex-solution"),
        gan=Path(REPO/"ex/gan-detector/ex-solution"),
        asd=Path(REPO/"ex/asd-diagnosis/ex-solution"),
        qml=Path(REPO/"ex/quantum-ml/ex-solution"),
        )

title = dict(
        lhcf="Tuesday/Reconstructing_LHCf_data",
        gan="Tuesday/Simulating_detectors_with_GANs",
        asd="Wednesday/ASD_diagnosis_with_MRI_and_radiomics",
        qml="Wednesday/Quantum_Machine_Learning",
        infra="Thursday/Environments",
        )


parser = ArgumentParser()
for ex in exercises.keys():
    parser.add_argument(f"--{ex}", action="store_true")

parser.add_argument("--list-users", action="store_true", help="List users and exits")

parser.add_argument("--solutions", action="store_true")
parser.add_argument("--force", action="store_true")

parser.add_argument("users", nargs="*")

args = parser.parse_args()

if args.list_users:
    users = []
    for user_data_path in USER_DATA_PATHS:
        for dirname in glob(str(user_data_path / "user-*")):
            users.append(re.findall("user-([\w\d-]+)", dirname)[-1])

    print (" ".join(sorted(set(users))))

    exit(0)

all_users = len(args.users)==0
all_ex = not any(getattr(args, ex) for ex in exercises.keys())

def my_copy(source, destination):
    if not destination.exists():
        print (f"{destination} does not exist. Pushing fresh copy!" )
        if source.is_dir():
            shutil.copytree(source, destination)
        elif source.is_file():
            os.makedirs(str(destination.parent), exist_ok=True)
            shutil.copy(source, destination)
    elif args.force:
        print (f"{destination} exists, --force overwriting" )
        if source.is_dir():
            shutil.rmtree(destination)
            shutil.copytree(source, destination)
        elif source.is_file():
            os.remove(destination)
            shutil.copy(source, destination)
    else:
              print (f"{destination} exists. Skipping.")


for user_data_path in USER_DATA_PATHS:
    for dirname in glob(str(user_data_path / "user-*")):
        user = re.findall("user-([\w\d-]+)", dirname)[-1]
        if not all_users and user not in args.users:
            continue
        print (f"Identified folder of user {user}")

        for ex, path in exercises.items():
            if not all_ex and not getattr(args, ex):
                continue
            else:
                print (f"Considering copying exercise {ex} to the {user}'s home")
                destination = Path(dirname) / title.get(ex, ex) if path.is_dir() else Path(dirname) / title.get(ex, ex) / path.name
                my_copy(path, destination)

        if not args.solutions:
            continue

        for ex, path in solutions.items():
            if not all_ex and not getattr(args, ex):
                continue
            else:
                print (f"Considering copying solution for {ex} to the {user}'s home")
                destination = Path(dirname) / title.get(ex, ex) / "solutions" if path.is_dir() else Path(dirname) / title.get(ex, ex) / "solutions" / path.name
                my_copy(path, destination)

