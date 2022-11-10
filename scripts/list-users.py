#!/bin/python

from subprocess import check_output
from argparse import ArgumentParser, RawTextHelpFormatter
import re

parser = ArgumentParser(
    formatter_class = RawTextHelpFormatter,
    description="""
`pattern` represents a Python-style regular expression used to match
domain names of remote machines.

Users are returned as a space-separated list.
""",
  epilog="""
  Example
  $ ./list-users.py cnaf
    user1 user2 user3
""")
parser.add_argument("pattern", nargs="*", help="Pattern to match machines")
args = parser.parse_args()

user_list = []
for machine in open('machines.dat').read().split('\n'):
  if machine == "": continue 

  if len(args.pattern)==0 or any([re.match(".*%s.*"%p, machine) for p in args.pattern]):
    privates = check_output(["ssh", '-Tx', machine, "ls /jupyter-mounts/users"])
    user_list += filter(lambda s: s != '', privates.split("\n"))

print (" ".join(set(user_list)))
