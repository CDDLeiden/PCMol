# PCMol 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Multi-target generative model for de novo drug design that utilizes latent embeddings of AlphaFold2 for conditioning.

![alt text](assets/PCMol.png)

## Requirements

- Python 3.9+
- rdkit 2021.03.5.0

## Installation


```bash
# Create conda env
conda env create -f environment.yml
conda activate pcmol
```

## Generating molecules for a particular target

```bash
# Run the model on a single target
python pcmol/generate.py --targets P21819
```
If available, the appropriate AlphaFold2 embeddings will be downloaded automatically. The generated molecules will be saved in the `data/results` folder.

## List of supported targets

The model currently depends on the availability of AlphaFold2 embeddings for the target protein. The list of supported targets can be found in the [data/targets.txt](data/targets.txt) file.

## Paper

The paper is available on [arXiv](https://arxiv.org/abs/2109.02019).