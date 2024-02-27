#!/bin/bash

# Conda
printf "\n\nSetting up the conda environment..."
conda env create -f environment.yml
conda activate pcmol
pip install -e .

# Download the pretrained model files
printf "\n\nDownloading the model files..."

cd data/models
wget https://surfdrive.surf.nl/files/index.php/s/T0wUBOmAEYYxxOo/download -O XL.tar
tar -xvf XL.tar
rm XL.tar
cd ../..

## Test the installation
printf "\n\nTesting the installation... Generating SMILES for P29275 (A2AR_HUMAN) using the pretrained model."

python pcmol/generate.py --target P29275