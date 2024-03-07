import os
import sys
from typing import Sequence

from tqdm import tqdm

from pcmol.utils.protein_sequence import ProteinSequence
from pcmol.utils.proteins import download_fasta, read_fasta


def load_protein_sequence(protein_id: int, data_folder: os.PathLike) -> ProteinSequence:
    """
    If needed download the .fasta file based on protein ID and return a
    ProteinSequence object
    """

    # Check if downloaded
    filename = f"{data_folder}/{protein_id}/{protein_id}.fasta"
    if not os.path.isfile(filename):
        download_fasta(protein_id, data_folder)

    protein_seq = read_fasta(filename)
    return protein_seq


def get_papyrus_proteins(
    papyrus_file, output_folder, start=0, end=61085165
) -> dict[int, ProteinSequence]:
    """
    Scans through the whole papyrus dataset, and downloads all required .fasta files from
    uniprot, based on protein accesion IDs (attribute[9] in papyrus)

    Returns a dict of ProteinSequence objects indexed via protein_ids
    """
    if not os.path.isfile(papyrus_file):
        print("Papyrus file not found.")
        sys.exit()

    end = 61085165 if not end else end

    with open(papyrus_file, "r") as papyrus:

        _ = papyrus.readline()
        proteins = {}

        for _ in tqdm(range(start, end)):
            entry = papyrus.readline()
            attributes = entry.split("\t")
            protein_id = int(attributes[9])

            if not proteins.get(protein_id):
                p_sequence = load_protein_sequence(
                    protein_id, data_folder=output_folder
                )
                proteins[protein_id] = p_sequence
            else:
                proteins[protein_id].n_ligands += 1
    return proteins


def get_proteins(
    pids: Sequence[int], output_folder: os.PathLike
) -> dict[int, ProteinSequence]:
    """
    Downloads all required .fasta files from
    uniprot, based on protein accesion IDs (attribute[9] in papyrus)

    Returns a dict of ProteinSequence objects indexed via protein_ids
    """
    proteins = {}
    for protein_id in tqdm(pids):
        p_sequence = load_protein_sequence(protein_id, data_folder=output_folder)
        proteins[protein_id] = p_sequence
    return proteins
