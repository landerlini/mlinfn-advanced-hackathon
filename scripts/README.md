# Scripts 

This folder includes scripts used to distribute the notebooks 
prepared for the hackathon to the students.

The list of machines and connection details is reported in `machines.dat`.

The script `prepare.sh` is automatically copied to the remote machines when `prepare.py` is used locally.

The script `list-users.py` produces a list of unique users with a folder in at least one of the machines. 

Finally, the script `prepare.py` is the admin entry point and is described in more detail in this README.

```
usage: prepare.py [-h] [--cleanup] [--solutions] [pattern [pattern ...]]
age: prepare.py [-h] [--cleanup] [--cleanup-da] [--cleanup-xai]
                  [--cleanup-gnn] [--cleanup-unet] [--solutions-da]
                  [--solutions-unet] [--solutions-gnn] [--solutions-xai]
                  [--user [USER [USER ...]]]
                  [pattern [pattern ...]]

Admin entry point to distribute notebooks through the computing nodes.

positional arguments:
  pattern               Pattern of Python-style regular expression used to match some or all the machines

optional arguments:
  -h, --help            show this help message and exit
  --cleanup             Reinitialize the user directories
  --cleanup-da          Reinitialize Domain Adaptation
  --cleanup-xai         Reinitialize Explainability
  --cleanup-gnn         Reinitialize GNN and transformers
  --cleanup-unet        Reinitialize the Lung Segmentation
  --solutions-da        Includes in the students directories the solutions to the Domain Adaptation part
  --solutions-unet      Includes in the students directories the solutions to the Lung Segmentation part
  --solutions-gnn       Includes in the students directories the solutions to the GNN and Transformers part
  --solutions-xai       Includes in the students directories the solutions to the Explainability part
  --user [USER [USER ...]], -u [USER [USER ...]]
                        force application to a specific user, ignoring user protections, and creating folders if missing

In the worst case scenario, chalk and blackboard...
```

## Scenarios

### During testing sessions
Admin and testers may want to restart the setup to the origin
```
./prepare.py .* --cleanup
```
This will ask a confirmation because if issued erroneously may destroy the work of users.

### Start up
While users connect their folders get created.
Admin can populate them with 
```
./prepare.py .* 
```

### Providing solutions
At the end of the hackathon you can provide solutions with 
```
./prepare.py .* --solutions-da --solutions-unet --solutions-xai --solutions-gnn
```

### A user needs the solution
You can run the command for a specific user on a specific machine, for example 
```
./prepare.py recas --solutions-da -u test_user
```
Adds the solution to the exercise on Domain Adaptation to the workspace of the
user named `test_user`.


### A user breaks a notebook and ask to restart from scratch
Ask the user to delete or rename the notebook and then renew it
```
./prepare.py .* -u <the_username>
```

### A user breaks a notebook
Ask the user to delete or rename the notebook and then renew it
```
./prepare.py .* -u <the_username>
```

### Five minutes before the beginning of an afternoon excercise, a bug is found
First, fix the bug in github.

Then, you can clean and reset the folder to a newer version of the notebook 
(e.g. on Explainability) with
```
./prepare.py * --clean-xai 
```
this avoids modifications to the notebooks in other folders 






