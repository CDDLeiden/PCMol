# PCMol 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Multi-target generative model for de novo molecule generation.

![alt text](assets/PCMol.png)

By using the internal protein representations of AlphaFold, a single trained model can generate relevant molecules for thousands of protein targets. 

![alt text](assets/targets.png)

## Requirements

- **Python** 3.8+
- **rdkit** 2021.03.5.0
- **Torch** 1.7 - 2.1

## Installation


```bash
# Setting up a fresh conda environment
conda env create -f environment.yml && conda activate pcmol
git clone https://github.com/andriusbern/pcmol.git && cd pcmol
python -m pip install -e .
```

## Pretrained model

The pretrained model can be downloaded from [here](https://drive.google.com/drive/folders/1-5Z3QZ3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3?usp=sharing). It should then be placed in the `.../pcmol/data/models` folder.

## Generating molecules for a particular target

### 1. Using a script
```bash
# Run the model on a single target using UniProt ID (generates 10 SMILES strings)
python pcmol/generate.py --target P21819
```

If available, the appropriate AlphaFold2 embeddings to be used as input to the model will be downloaded automatically. The generated molecules will be saved in the `data/results` folder.

### 2. Calling the generator directly
```python
# Run the model on a single target using a custom embedding
from pcmol import Generator

generator = Generator(model="XL")
SMILES = Generator.generate_smiles(target="P21819", num_mols=100)
```

## List of supported targets

The model currently depends on the availability of AlphaFold2 embeddings for the target protein. The list of supported targets can be found in the [data/targets.txt](data/targets.txt) file.

## Training

To retrain the model you first need to download the dataset from [here]().

```bash
## Train the model
python pcmol/train.py --model default
```

---

## Paper & Authors

To be added.