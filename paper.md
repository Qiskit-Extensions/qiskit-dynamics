---
title: 'Qiskit Dynamics: A Python package for simulating the time dynamics of quantum systems'
tags:
  - Python
  - quantum
  - quantum computer
  - pulse
  - control
authors:
  - name: Daniel Puzzuoli
    corresponding: true
    affiliation: 1
  - name: Christopher J. Wood
    affiliation: 2
  - name: Daniel J. Egger
    affiliation: 3
  - name: Benjamin Rosand
    affiliation: 4
  - name: Kento Ueda
    affiliation: 5
affiliations:
 - name: IBM Quantum, IBM Canada, Vancouver, BC, Canada
   index: 1
 - name: IBM Quantum, IBM T.J. Watson Research Center, Yorktown Heights, NY, USA
   index: 2
 - name: IBM Quantum, IBM Research Europe - Zurich, Ruschlikon, Switzerland
   index: 3
 - name: Yale University, New Haven, CT, USA
   index: 4
 - name: IBM Quantum, IBM Research Tokyo, Tokyo, Japan
   index: 5


date: 13 September 2023
bibliography: paper.bib
---

# Summary

Qiskit Dynamics is an open source Python library for numerically simulating the time dynamics of finite-dimensional quantum systems. The goal of the package is to provide flexible configuration of the numerical methods used for simulation: general tools for transforming models of quantum systems for more efficient simulation (rotating frames and the rotating wave approximation), choice of array representations (dense vs. sparse, and different array libraries), and access to different types of underlying solvers (standard ODE vs. geometric solvers). The package also contains advanced functionality for computing time-dependent perturbation theory expressions used in robust quantum control optimization [@perturb1; @perturb2].

As part of the Qiskit Ecosystem (https://qiskit.org/ecosystem), the package interfaces with other parts of Qiskit [@Qiskit]. Most notably, Qiskit Dynamics provides tools for simulating control sequences specified by Qiskit Pulse [@alexander_qiskit_2020] which gives a pulse representation of quantum circuit instructions. Higher level interfaces allow users to build and interact with simulation-based objects that target the same constraints (qubit layout, control sequence sampling rate, etc.) as a specified IBM Quantum computer.

Lastly, to facilitate high-perfomance applications, Qiskit Dynamics is compatible with the JAX array library [@jax2018github]. As such, all core computations are just-in-time compilable, automatically differentiable, and executable on GPU.

# Statement of need

Numerical simulation of time-dependent quantum systems is a useful tool in quantum device characterization, design, and control optimization. As these applications often involve the expensive process of repeatedly simulating a system across different parameters (e.g., in exploratory parameter scans or in optimizations), users need to be able to easily select the numerical methods that are most performant for their specific problem. The ability to automatically differentiate and compile simulations is also critical for control optimization research. To ensure flexibility and broad applicability, it is important that all of these capabilities work for arbitrary user-specified models.

Furthermore, having a simulation-based drop-in replacement for real quantum computing systems is useful for developers building software tools for low-level control of experiments, such as Qiskit Pulse [@alexander_qiskit_2020] and Qiskit Experiments [@kanazawa_qiskit_2023].

# Related open source packages

Due to its importance, many open source packages contain time-dependent quantum system simulation tools. In Python, these include QuTiP [@qutip], TorchQuantum [@torchquantum], and C3 [@C3]. C++ packages (also with Python interfaces) include lindbladmpo [@lindbladmpo] and Quandary [@quandary]. Packages also exist in other languages, such as the Hamiltonian open quantum system toolkit (HOQST) [@hoqst] and a Framework for Quantum Optimal Control [@julia_qc] in Julia, and Spinach [@spinach] in MATLAB. The features in Qiskit Dynamics for simulating Qiskit Pulse control sequences replace those previously offered in Qiskit Aer [@aer].

# Documentation and community

Qiskit Dynamics documentation, including API docs and tutorials, is available at https://qiskit.org/ecosystem/dynamics/. A public slack channel for community discussion can be found here https://qiskit.slack.com/archives/C03E7UVCDEV.

# Acknowledgements

We would like to thank Ian Hincks, Naoki Kanazawa, Haggai Landa, Moein Malekakhlagh, Avery Parr, R. K. Rupesh, William E. Shanks, Arthur Strauss, Matthew Treinish, Helena Zhang for helpful discussions, reviews, and contributions to the package.

# References
