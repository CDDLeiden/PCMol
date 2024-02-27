#!/bin/bash

# Conda
printf "\n\nSetting up the conda environment..."
conda env create -f environment.yml
eval "$(conda shell.bash hook)"
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
printf "\n\nTesting the installation... Generating SMILES for P29275 (A2AR_HUMAN) using the pretrained model.\n"

python pcmol/generate.py --target P29275

printf "**************"
printf "\n\nInstallation complete. You can now use the package by activating the conda environment using \n\tconda activate pcmol"
printf "\n\nTo generate SMILES for a target protein, use the following command:"
printf "\n\tpython pcmol/generate.py --target <accession_ID>"
printf "\n\nFor example, to generate SMILES for P29275 (A2AR_HUMAN), use the following command:"
printf "\n\tpython pcmol/generate.py --target P29275"

printf "\n\n*If no GPU is available, the generation process may take a long time. Additional flag must be used when generating without GPU:"
printf "\n\tpython pcmol/generate.py --target P29275 --device cpu\n"