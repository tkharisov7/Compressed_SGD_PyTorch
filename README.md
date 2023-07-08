# Code guidelines

This implementation is based on PyTorch (1.5.0) in Python (3.8). 

It enables to run simulated distributed optimization with master node on any number of workers based on [PyTorch SGD Optimizer](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD) with gradient compression. Communication can be compressed on both workers and master level. Error-Feedback is also enabled. 

This is a fork of https://github.com/SamuelHorvath/Compressed_SGD_PyTorch

### Installation

To install requirements
```sh
$ pip install -r requirements.txt
```

###  Example Notebook
To run our code see [example notebook](example_notebook.ipynb).

### Theory

If you are interested in theoretical results, you may check the keynote files in the [theory folder](theory). 

### License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
