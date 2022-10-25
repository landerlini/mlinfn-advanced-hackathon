# ML_INFN Advanced Hackathon: Advanced Jupyter 
This repository includes the notebooks and a Snakefile used to provide an example 
on chaining multiple notebooks in a single workflow using Snakemake. 
Contents were developed for the Advanced Hackathon of ML_INFN organized in Bari
in November 2022. 

### Introduction
Jupyter notebooks are usually conceived as atomic work unit containing in their 
code blocks the whole code necessary to study or prototype some specific problem.
In this *hands-on* lecture, we will discuss  a set of tools to build simple 
workflows out of multiple Jupyter notebooks, focussing on specific aspects of the 
overall task. 
In particular, we will discuss:
 * how to access with a real terminal to INFN Cloud instances,
 * how to execute a jupyter notebook as a script, storing the graphical output
 * how to pass arguments to the jupyter notebooks when executed in batch mode
 * how to describe dependency relations between different notebooks, builing 
   automatic workflows running in batch on the INFN Cloud instance. 

A complete example workflow is provided, taking a typical HEP application as a use case.
Comparing how the production cross-section in proton-proton collisions 
of two charmonium states depends on the transverse momentum. 
We will focus on $J/\psi$ and $\psi(2S)$ states, produced in CMS and reconstructed 
in a muon pair. 
We will use the dataset collected in 2011 and made available as part of the 
[CERN OpenData](https://opendata.cern.ch/record/5201).

For those not familiar with HEP, here is what you really need to know:
 * we obtain a dataset that contains a variable M (mass) and variables that allows to 
   compute a variable PT (momentum)
 * in the mass distribution, we can observe two peaks. A large peak at M ~ 3.1 GeV/$c^2$,
   and a smaller peak at M ~ 3.7 GeV.
 * we want to count the number of entries in each peak, in bins of transverse momentum PT.

We expect the ratio of the two counts to be constant is $J/\psi$ and $\psi(2S)$ are 
similar objects (and they are) with similar production mechanisms (and they have). 

Forgetting everything about the physics, this exercise makes it possible 
to describe a workflow in which we develop some complex analysis in 
a jupyter notebook (in this case the extended maximum likelihood fit used
to count the number of entries in each peak) under some assumptions
(in this case we will develop the fit on the whole dataset) to then
apply it programmatically under difference conditions (in this case, 
in different bins). 

The exercise is then composed of three notebooks:
 * the main analysis notebook
 * a notebook splitting the dataset in transverse momentum bins 
 * a notebook taking the results obtained in each bin and plotting the dependence

The workflow will be defined in a Snakefile file, defining who is the input 
and who is the output for each step, as we will discuss at the end.
   

### Prerequisites 
In order to use this repository on ML_INFN instances of INFN Cloud, 
some additional packages are need and can be installed with aptitude 
and pip.
```
apt-get update; apt-get install vim tmate snakemake screen; pip iminuit
```

### Executing notebooks in batch with `tmate` and `nbconvert`
In order to pipe multiple notebooks in a workflow, the first step is to acquire 
the skills to execute the notebook in a programmatic way, without accessing 
the Jupyter user interface. 
Now, the terminal made available by Jupyter lacks of some crucial features, as 
font colors, reverse search, and even recalling previous commands with the 
keypad arrows is not possible. 
To be able to use a decent terminal running within the INFN images we will 
use `tmate`. `tmate` is a small program that connects to a remote server 
and provides a secure ssh access to a virtual terminal instance through 
that remote server. 

To launch tmate, just open a terminal in Jupyter and type 
```bash
tmate
```
you should see a welcome message appearing, including an ssh command that you 
can copy-paste to a terminal on your device. 
You will then access, as root, to the ML_INFN instance, with the power of 
a real Linux terminal. 

Now, you can execute a notebook programmatically as
```bash
jupyter nbconvert <notebook_file> --execute --inplace
```
using the `--inplace` flag, the new output will overwrite any existing output 
obtained while editing the notebook via the graphical user interface, and 
this may be tedious. 
In this preferable to copy the notebook to another file and execute it 
there, leaving the original file untouched. 

```bash
jupyter nbconvert <notebook_file> --execute --output <new_notebook_file>
```


### Passing variables to the notebook
To steer the behaviour of a notebook, while running it programmatically, 
it is useful to define some mechanism to transfer variables from bash 
to python. Of course, one can write a file from bash and read it back 
from Python, but this may lead to difficult-to-debug situations if 
we plan to run multiple instances of the notebook in parallel. 

A safer solution is to rely on envioronmental variables. In bash, 
one can set a variable for a single process with the syntax
```bash
MY_VAR=ITS_VALUE ANOTHER_VAR=ITS_VALUE mycommand its_args;
```

Hence, if we want for example to pass the value 42 for the 
random seed used in our notebook while running it programmatically, 
we can type
```bash
MY_SEED=42 jupyter nbconvert <notebook_file> --execute --output <new_notebook>
```
to get that variable from one of the code blocks of the notebook,
we will need to write
```python
import os;
my_seed = int(os.environ.get("MY_SEED", 0))
```
where you should note that:
 * environment variables are strings, if you need an integer, you 
   should provide an explicit cast
 * when running the notebook interactively, the variable "MY_SEED" won't be set,
   so you should define the default value (in the example, 0) to use.
 
### Creating workflows of notebooks with Snakemake
Now that we know how to programmatically execute notebooks and how to steer
their behaviour, we are ready to define a sequence of notebooks that 
we want to be executed one after the other.
If the sequence is simple enough, a bash script can be sufficient, but 
for more complex or time-consuming tasks, where parallel execution may
play a role, bash is suboptimal. 

Instead of defining a sequence of tasks, we aim at describing the 
problem as a set of dependencies. To run notebook A, we will need 
the output of notebooks B and C, that in turn will need to load 
some input data file. 
With such a structure, when we wish to update the result of A, 
we can simply check which of the dependencies on disk need to 
be updated, parallelizing if needed the execution of the 
notebooks B and C, and at the end rerunning A.

A very famous application used to build dependency graphs like these
is GNU Make. 
Unfortunately, GNU Make has its own language which has been designed 
specifically to address dependency problems between software building 
applications, such as compilers and linkers, and is not particularly 
user friendly.
A more recent software package, named 
[Snakemake](https://snakemake.readthedocs.io/en/stable/), tries to combine 
the idea of describing the dependency graph of a workflow, with the 
simplicity and versatility of the Python language. 

Providing a complete overview of the Snakemake capabilities is 
well beyond the scope of this README and package, but an example 
Snakefile defining a workflow of the three notebooks described in 
the introduction is provided. 

Once defined, a Snakefile can be executed simply typing 
```bash
snakemake <targets>
```
where `targets` is a space-separated list of rules or output that we 
wish to regenerate. 

With the Snakefile provided in this package, you can try typing
```bash
snakemake collect
```
to run the whole pipeline. 
If immediately after running it, you try to run it again, you will 
see no effect, as all of the input files are found unchanged since 
last run.
With appropriate arguments, one can force the execution of the whole
workflow (`-F`) or of the most downstream rule (`-f`). 

Snakemake deals with parallel execution on multiple cores, assuming 
that the execution of each notebook will occupy one core. 
The number of parallel jobs can be defined by the user with 
`-j <number_of_concurrent_jobs>`. For example,
```bash
snakemake collect -Fj1
```
will force the execution of the whole workflow running one 
notebook at a time.




