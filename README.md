# Scalable Hierarchical Graph Generation via Soft Community Structure

Source code for the paper **"Scalable Hierarchical Graph Generation via Soft Community Structure."**, **Schema**

Python version: 3.12.3 (also tested with 3.13.5). 

## Installation

We strongly recommend using a virtual environment, either with `venv` or `conda`.


### Create a virtual python environment: 

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### CUDA

We used CUDA 12.9 for our experiments. Please select the appropriate PyTorch build for your CUDA version if you are using a different version.

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129

pip install pyg_lib==0.6.0 torch_scatter==2.1.2 torch_sparse==0.6.18 torch_cluster==1.6.3 torch_spline_conv==1.2.2 -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
```


### CPU

If you have a CPU-only environment, install CPU versions of the packages above:

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu

pip install pyg_lib==0.6.0 torch_scatter==2.1.2 torch_sparse==0.6.18 torch_cluster==1.6.3 torch_spline_conv==1.2.2 -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
```

### Other Packages

```bash
pip install torch_geometric==2.7.0
pip install pandas==3.0.3 leidenalg==0.12.0 igraph==1.0.0 scipy==1.17.1 ogb==1.3.6
```

#### IGB Dataset
```bash
pip install git+https://github.com/IllinoisGraphBenchmark/IGB-Datasets.git
pip install colorama==0.4.6
```

**Download the package after installation**
```bash
from igb import download
download.download_dataset(path='./data/igb', dataset_type='homogeneous', dataset_size='medium')
```

#### Evaluation Packages

```bash
pip install scikit-learn==1.9.0 orbit-count==0.1.0
pip install omegaconf==2.3.1 tabulate==0.10.0
```

#### Baselines

Due to cross-package dependencies, we used individual conda environments for the baselines. For installation, follow the original repositories.


## Project

The project consists of three main directories: `clustering`, `generation`, and `evaluation`.

Each directory contains its own `main.py` and argument configuration. Please refer to the respective directory for usage details.

Here you can find the basic usage:

<TODO>