# PCMol 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A multi-target model for de novo molecule generation. By using the internal protein representations of the AlphaFold model, a single SMILES-based transformer can generate relevant molecules for thousands of protein targets. 

![alt text](assets/PCMol.png)



![alt text](assets/targets.png)

<!-- ## Requirements

- **Python** 3.8+
- **rdkit** 2021.03.5.0+
- **Torch** 1.7 - 2.1 -->

---

## Installation

The model can be used in two different ways: either by using the provided docker image or by setting up a conda environment. 

### 1. Docker

*Note: The docker image is currently not available.*

The docker image contains all the prerequisites and the pretrained model.
```bash
# Pull the docker image
docker pull andriusbern/pcmol:latest
```

### 2. Conda
The conda route requires the user to download the pretrained model and training sets manually.

```bash
# Setting up a fresh conda environment
conda env create -f environment.yml && conda activate pcmol
git clone https://github.com/andriusbern/pcmol.git && cd pcmol
python -m pip install -e .
```

---

## Pretrained model

When using the conda route, download the pretrained model (and training sets) directly from the command line, run the following command:
```bash
python download.py --model XL ## Pulls the model file and training sets
```

**Alternatively, the pretrained model can be downloaded from [here](https://drive.google.com/drive/folders/1-5Z3QZ3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3?usp=sharing). It should then be placed in the `.../pcmol/data/models` folder.*



## Generating molecules for a particular target

### 1. Docker
```bash
# Run the model on a single target using UniProt ID (generates 10 SMILES strings)
docker run -it andriusbern/pcmol:latest python -m pcmol.generate --target P21819
```

### 2. Using a script (conda route)
```bash
# Run the model on a single target using UniProt ID (generates 10 SMILES strings)
python pcmol/generate.py --target P21819
```

If available, the appropriate AlphaFold2 embeddings to be used as input to the model will be downloaded automatically. The generated molecules will be saved in the `data/results` folder.

### 3. Calling the generator directly

To generate molecules for a particular target, the `Runner` class can be used directly. The `generate_smiles` method returns a list of SMILES strings for a target protein specified by its UniProt ID.
```python
from pcmol import Runner

model = Runner(model="XL")
SMILES = model.generate_smiles(target="P21819", num_mols=100)
```

## List of supported protein targets

The model currently depends on the availability of AlphaFold2 embeddings for the target protein. The list of supported targets can be found in the [data/targets.txt](data/targets.txt) file.

<!-- ## Training

To retrain the model you first need to download the dataset from [here]().

```bash
## Train the model
python pcmol/train.py --model default
``` -->

---

## Paper & Authors

To be added.