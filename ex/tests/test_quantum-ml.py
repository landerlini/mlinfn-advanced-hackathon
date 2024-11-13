import pytest

###########################################################################

def test_torch():
    import torch

    torch.cuda.init()
    assert torch.cuda.is_available()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    rnd = torch.randn(size=(100, 1)).to(device)
    assert torch.abs(torch.sum(rnd)) > 0.0


def test_pennylane():
    import pennylane as qml

    dev = qml.device("lightning.gpu", wires=2)

    @qml.qnode(dev)
    def circuit():
      qml.Hadamard(wires=0)
      qml.CNOT(wires=[0,1])
      return qml.expval(qml.PauliZ(0))

    res = circuit()
    assert res == 0.0


def test_qiskit():
    import qiskit
    from qiskit_aer import AerSimulator
    
    # Generate 3-qubit GHZ state
    circ = qiskit.QuantumCircuit(3)
    circ.h(0)
    circ.cx(0, 1)
    circ.cx(1, 2)
    circ.measure_all()

    # Construct an ideal simulator
    aersim = AerSimulator()

    # Perform an ideal simulation
    result_ideal = aersim.run(circ).result()
    counts_ideal = result_ideal.get_counts(0)
    assert isinstance(counts_ideal, dict)


def test_all_imports():
    # QClassifier_PennylanePytorch
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from time import time
    from tqdm import tqdm
    
    import torch
    import pennylane as qml
    
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    
    from torch.autograd import Variable

    # QAE_PennylanePytorch
    import matplotlib.pyplot as plt
    import os
    from time import time
    from tqdm import tqdm

    import torch
    from torch.autograd import Variable
    import pennylane as qml
    from pennylane import numpy as np

    from IPython.display import Image

    # QUBO_GraphColoring
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    import itertools
    import argparse

    from qiskit.quantum_info import Pauli, SparsePauliOp
    from qiskit.circuit.library import TwoLocal
    from qiskit_algorithms.optimizers import SPSA
    from qiskit.primitives import Sampler
    from qiskit_algorithms import SamplingVQE

    from qiskit_optimization import QuadraticProgram
    from qiskit_algorithms import NumPyMinimumEigensolver
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
