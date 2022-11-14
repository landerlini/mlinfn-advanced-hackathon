#!/bin/python3
from subprocess import check_output
from argparse import ArgumentParser, RawTextHelpFormatter
import re

parser = ArgumentParser(
    formatter_class = RawTextHelpFormatter,
    description="Admin entry point to distribute notebooks through the computing nodes.",
    epilog="""In the worst case scenario, chalk and blackboard...""")

### Arguments
parser.add_argument(
    "pattern", 
    nargs="*", 
    help="Pattern of Python-style regular expression used to match some or all the machines"
  )

parser.add_argument(
    "--cleanup", 
    action='store_true', 
    help="Reinitialize the user directories"
    )

parser.add_argument(
    "--cleanup-da", 
    action='store_true', 
    help="Reinitialize Domain Adaptation"
    )

parser.add_argument(
    "--cleanup-xai", 
    action='store_true', 
    help="Reinitialize Explainability"
    )

parser.add_argument(
    "--cleanup-gnn", 
    action='store_true', 
    help="Reinitialize GNN and transformers"
    )

parser.add_argument(
    "--cleanup-unet", 
    action='store_true', 
    help="Reinitialize the Lung Segmentation"
    )

parser.add_argument(
    "--solutions-da", 
    action='store_true', 
    help="Includes in the students directories the solutions to the Domain Adaptation part"
    )

parser.add_argument(
    "--solutions-unet", 
    action='store_true', 
    help="Includes in the students directories the solutions to the Lung Segmentation part"
    )

parser.add_argument(
    "--solutions-gnn", 
    action='store_true', 
    help="Includes in the students directories the solutions to the GNN and Transformers part"
    )

parser.add_argument(
    "--solutions-xai", 
    action='store_true', 
    help="Includes in the students directories the solutions to the Explainability part"
    )

parser.add_argument(
    "--user", "-u",
    nargs="*",
    help="force application to a specific user, ignoring user protections, and creating folders if missing"
    )

args = parser.parse_args()

protected_users = []
#  'afania', 
#  'giorgia-20miniello',
#  'spiga'
#]

user_list = []
for machine in open('machines.dat').read().split('\n'):
  if machine == "": continue 

  if len(args.pattern)==0 or any([re.match(".*%s.*"%p, machine) for p in args.pattern]):
    ## Determine the user list
    user_list = args.user
    if user_list is None or len(user_list) == 0:
      privates = str(check_output(["ssh", '-Tx', machine, "ls /jupyter-mounts/users"]), 'ascii')
      user_list = list(filter(lambda s: (s != '' and s not in protected_users), privates.split("\n")))

    print ("Users:\n" + "\n".join([' - '+u for u in user_list]))

    if any([getattr(args, 'cleanup'+nb) for nb in ['', '_da', '_unet', '_gnn', '_xai']]):
      response = input(f"You are potentially destroying the work of listed users on "
                       f"{machine[machine.index('@')+1:]}. \nTo continue, type: destroy > ")
      if response != 'destroy':
        print ("Aborted.")
        continue 


    check_output(["scp", "./prepare.sh", f"{machine}:~/prepare.sh"])

    shell_args = []
    shell_args += ['--cleanup'] if args.cleanup else []
    shell_args += ['--solutions-da'] if args.solutions_da else []
    shell_args += ['--solutions-unet'] if args.solutions_unet else []
    shell_args += ['--solutions-gnn'] if args.solutions_gnn else []
    shell_args += ['--solutions-xai'] if args.solutions_xai else []
      
    check_output(["ssh", '-Tx', machine, "sh prepare.sh"] + shell_args + user_list)


