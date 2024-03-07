"""
Script for running AlphaFold on protein sequences present in papyrus
"""

import os, sys, random
import subprocess, shutil
import pickle, array
import numpy as np
from papyrus import get_papyrus_proteins

join = os.path.join

### DSLAB CONFIG
UID = random.randint(0, 10000)
UID = 0
DATA_DIR = "/data/bernataviciusa/af_data"
FASTA_DIR = join(DATA_DIR, "fasta")
PAPYRUS_PATH = join(DATA_DIR, "papyrus.tsv")
OUT_DIR = join(DATA_DIR, "af_output", str(UID))
ALPHAFOLD_DIR = "/data/bernataviciusa/alphafold-2.2.0"
DATESTRING = "2022-04-14"


def filter_proteins(protein_dict, max_len=450):
    """
    Returns a sorted list of protein_ids, sorted by number of datapoints in papyrus
    """

    # Length cut-off
    prot_list = []
    for protein in protein_dict.values():
        if protein:  # Filter out None vals
            if len(protein) < max_len:
                prot_list.append(protein)

    # Sort by number of datapoints
    order = np.argsort([protein.n_ligands for protein in prot_list])
    sorted_proteins = [prot_list[i] for i in order[::-1]]

    return sorted_proteins


def run_alphafold(protein_id=None, fasta_input=None, uid=0):
    """
    Run AlphaFold on either a protein_id (will try to scan the dir) or the path
    to .fasta file
    """

    if protein_id:
        fasta_input = join(FASTA_DIR, f"{protein_id}.fasta")

    command = f"bash run_alphafold.sh -d {DATA_DIR} -o {OUT_DIR} -f {fasta_input} -t \
        {DATESTRING} -c reduced_dbs -p true"
    subprocess.Popen(command)

    # Duplicate the input fasta file at AlphaFold's output dir
    fasta_copy = join(OUT_DIR, protein_id, f"{protein_id}.fasta")
    shutil.copy(fasta_input, fasta_copy)


def batch_process(start=0, end=-1):
    """
    Run AlphaFold on a list of protein ids
    """

    protein_ids = get_papyrus_proteins(PAPYRUS_PATH, output_folder=FASTA_DIR)

    end = len(protein_ids) if end == -1 else end

    # for i, protein in enumerate(protein_ids):
    for i in range(start, end):

        protein_id = protein_ids[i]
        print(
            f'\n{"*"*100}\n  Running protein {protein_id}... {i+1}/{end}\n{"*"*100}\n'
        )

        if not check_if_already_processed(protein_id):
            run_alphafold(protein_id)
            clean_alphafold_output(protein_id)

        else:
            print(f"AlphaFold output for {protein_id} already exists.")


def clean_alphafold_output(protein_id):
    """
    Sorts the .pkl and .pdb files; extracts the relevant internal representations and
    stores them as .npy files
    """
    try:
        prot_dir = join(OUT_DIR, protein_id)

        process_result_pkl(prot_dir)

        files = os.listdir(prot_dir)
        pdb_dir = join(prot_dir, "pdb")
        pkl_dir = join(prot_dir, "pkl")
        os.mkdir(pdb_dir)
        os.mkdir(pkl_dir)

        for output_file in files:
            if output_file.endswith(".pdb"):
                shutil.move(join(prot_dir, output_file), pdb_dir)
            elif output_file.endswith(".pkl"):
                shutil.move(join(prot_dir, output_file), pkl_dir)
    except:
        print(f"Could not clean up AF output for protein {protein_id}.")


def check_if_already_processed(protein_id):

    af_out_dir = join(OUT_DIR, protein_id)
    if os.path.exists(af_out_dir):
        files_to_check = [
            "result_model_5_pred_0.pkl",
            "ranked_4.pdb",
            "relaxed_model_5_pred_0.pdb",
        ]

        exists = [os.path.isfile(join(af_out_dir, f)) for f in files_to_check]
    return all(exists)


def process_result_pkl(directory):
    """
    Store evoformer and structure module representations in protein_id/representations
    folder
    """
    for num in range(1, 6):

        path = join(directory, "result_model_{num}_pred_0.pkl")
        pkl = open(path, "rb")
        result = pickle.load(pkl)

        reps = result["representations"]
        single, struct = reps["single"], reps["structure_module"]

        rep_dir = join(directory, "representations")
        os.mkdir(rep_dir)

        np.save(join(rep_dir, f"single_{num}.npy"), single)
        np.save(join(rep_dir, f"struct_{num}.npy"), struct)


def test():
    protein_dict = get_papyrus_proteins(PAPYRUS_PATH, output_folder=FASTA_DIR)
    filtered = filter_proteins(protein_dict)

    print(filtered)
    print(len(filtered))


if __name__ == "__main__":
    import argparse

    test()
