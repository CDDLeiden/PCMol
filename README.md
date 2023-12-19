# PCMol 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Multi-target generative model for de novo drug design that utilizes latent protein embeddings of AlphaFold2 for conditioning.

![alt text](assets/PCMol.png)

## Requirements

- Python 3.9+
- rdkit 2021.03.5.0
- Torch 1.7 - 2.1

## Installation


```bash
# Setting up a fresh conda environment
conda env create -f environment.yml && conda activate pcmol
git clone https://github.com/andriusbern/pcmol.git && cd pcmol
python -m pip install -e .
```

## Pretrained model

The pretrained model can be downloaded from [here](https://drive.google.com/drive/folders/1-5Z3QZ3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3?usp=sharing). It should then be placed in the `pcmol/data/trained_models` folder.

## Generating molecules for a particular target


```bash
# Run the model on a single target using UniProt ID
python pcmol/generate.py --targets P21819
```
If available, the appropriate AlphaFold2 embeddings will be downloaded automatically. The generated molecules will be saved in the `data/results` folder.

```python
# Run the model on a single target using a custom embedding
from pcmol.model.runner import Runner

## List of supported targets

The model currently depends on the availability of AlphaFold2 embeddings for the target protein. The list of supported targets can be found in the [data/targets.txt](data/targets.txt) file.

## Training

To retrain the model you first need to download the dataset from [here]().

```bash
## Train the model
python pcmol/train.py --model default
```

## Paper & Authors

To be added.