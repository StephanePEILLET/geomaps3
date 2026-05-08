[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# Geomaps3

Geomaps3 stands for Object Delineation on Earth Observations with Neural network.
It is a set of command-line tools performing semantic segmentation on remote
sensing images (aerial and/or satellite) with as many layers as you wish.

## Features
- **Architectures**: U-Net (Light, Medium, Large), ResNet-backed U-Net, ResBlockNet, DeeplabV3+, and dynamic models via Segmentation Models PyTorch.
- **Advanced Training**: Cyclic learning rates, customizable loss functions (BCE, Focal, Combo), extensive data augmentations, and weighted class sampling to handle class imbalances.

## Installation

These instructions assume that you already have [conda](https://conda.io/) installed.

First, download and extract a copy of geomaps from [repository](https://gitlab.com/dai-projets/geomaps-landcover).
Then navigate to the root of the geomaps directory in a terminal and run the following:

```bash
# Clone repository
git clone git@gitlab.com:dai-projets/geomaps-landcover.git
or
git clone https://gitlab.com/dai-projets/geomaps-landcover.git
or
download a release at https://gitlab.com/dai-projets/geomaps-landcover/-/releases

# Go to the root project folder
cd geomaps-landcover

# Install the environment
conda env create --file=environment.yml

# Activate the environment
conda activate geomaps

# Install snorkel in the environment
pip install .
```
## Documentation
You can find the documentation of the project at [https://dai-projets.gitlab.io/geomaps-landcover/](https://dai-projets.gitlab.io/geomaps-landcover/)

## Quickstart & Usage

Geomaps3 toolkit is fully driven by **Hydra**, which means you do not use JSON files anymore. You can configure and launch your training and inference pipelines natively from the command line or via YAML configurations located in `geomaps/configs/`.

To launch a training task, simply use the `geomaps` command with the `task=train` argument:

```bash
# Launch a basic training using default configuration
geomaps task=train

# Override configuration dynamically from the command line
geomaps task=train model=deeplab datamodule.batch_size=8 trainer.max_epochs=150

# Use a specific sampling file for class imbalance
geomaps task=train datamodule.sampling_file="/path/to/weights.csv"
```

### Configuration Structure
The project configuration relies on strict, structured Pydantic/Dataclasses ensuring type-safety. You can find the base schemas in `geomaps/configs/structured/`.

For more detailed information about models, data preparation, and training loops, please refer to the `docs/` folder or the online documentation.
