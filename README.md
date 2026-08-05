# Free Fermionic Line Graph States

See [algorithm_desc]. We provide some details and some code on how to prepare eigenstates
corresponding to a free fermionic Hamiltonian based on a line graph solution. There is
also some code on finding (largish) line graphs in general graphs.

## General Notes about the Code Base

This is essentially a part of a larger project (WIP) with a real manuscript that is
not public yet. Therefore, there won't be much development on this repo here, and the code
is not very clean as I do not intend to make it a library (it helps to read
[algorithm_desc] first). Nonetheless, I hope it will be useful as a starting reference.

The code is written in Python as we use some python libraries so that we do not have to
implement everything from scratch. Performance is not the goal here, but rather keeping
the algorithms easy to understand (you'll probably find some redundant work that is
executed by the code). Also, the code is probably not very pythonic; writing clean code in
python with proper typing (annotations) is just messy

## Environment Setup

The following is not perfect at all, but it works best for me during the initial
development. Maybe I'll set up a container...

### About SageMath

The code uses sagemath. I'm unable to install sagemath directly into a virtual python
environment, so until I set up a container, the code requires the sagemath library to be
installed (on the system or your own container); as well as numpy and matplotlib, but they
come with sagemath.

### Creating the Environment

- Once sagemath is installed, do something like `python -m venv .venv
  --system-site-packages` to get the system packages into the environment.
- Then activate it and do `pip install -r requirements.txt` to install the rest. (note
  that the requirements file does only list the top-level packages directly installed
  using pip; i.e., it's not useful for reproducibility).
- Then install the rust backend by either doing `pip install rust_backend/py_lib` or by
  first building the wheel manually and then installing it.


## License

The Pauli Tracker project is distributed under the terms of both the MIT license and the
Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT).

[algorithm_desc]:
https://github.com/taeruh/free_fermionic_line_graph_states/blob/main/algorithm_desc.pdf
