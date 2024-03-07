import csv
import os
from collections import OrderedDict
from urllib import request

import numpy as np

from pcmol.utils.protein_sequence import ProteinSequence


def get_names(
    target_proteins: list[int], folder: os.PathLike
) -> tuple[OrderedDict[int, str], OrderedDict[int, str]]:
    """
    Returns a dictionary of protein names for the target proteins (from Fasta headers)
    """
    proteins = [load_protein_sequence(pid, folder) for pid in target_proteins]
    headers = [p.info for p in proteins]
    name_strings = [h.split("|")[2].split("OS")[0] for h in headers]
    tags = [n.split(" ")[0] for n in name_strings]
    names = [" ".join([n.split(" ")[1:]]) for n in name_strings]
    target_proteins.sort()
    tag_dict = OrderedDict(dict(zip(target_proteins, tags)))
    name_dict = OrderedDict(dict(zip(target_proteins, names)))

    return name_dict, tag_dict


def download_fasta(protein_id: int, output_folder: os.PathLike) -> None:
    """
    Retrieves the fasta file from Uniprot based on protein ID
    """
    try:
        remote_url = f"https://uniprot.org/uniprot/{protein_id}.fasta"
        local_file = f"{output_folder}/{protein_id}.fasta"
        open(local_file, "a").close()
        request.urlretrieve(remote_url, local_file)
    except:
        print(f"Fasta file for {protein_id} could not be downloaded.")


def read_fasta(file_path: os.PathLike) -> ProteinSequence | None:
    """
    Reads a .fasta file and returns a Protein Sequence object
    """
    try:
        with open(file_path, "r") as fasta:
            content = fasta.readlines()
            header = content[0]
            sequence = content[1:]
            stripped = [line.strip("\n") for line in sequence]
            sequence = "".join(stripped)
            protein_id = header.split("|")[1].strip()

        protein_seq = ProteinSequence(
            sequence=sequence, protein_id=protein_id, info=header
        )
        return protein_seq
    except OSError:
        return None


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


def get_families(
    papyrus_targets_file: os.PathLike, targets: list[str]
) -> OrderedDict[str, int]:
    with open(papyrus_targets_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        _ = next(reader)
        families = []
        target_families = {}
        for line in reader:
            target = line[0].strip("_WT")
            classes = line[5].split("->")
            family = classes[1] if len(classes) > 1 else classes[0]
            if family == "":
                family = "Unknown"
            families += [family]
            target_families[target] = family

    families = list(set(families))
    # Create a dictionary of families and their members

    family_members = {}
    for protein in targets:
        family = target_families[protein]
        if family_members.get(family) is None:
            family_members[family] = [protein]
        else:
            family_members[family] += [protein]

    # Rename some families
    modified_families = OrderedDict()
    modified_families["Other"] = []
    modified_families["G protein-coupled receptor"] = []
    cutoff = 25  # Number of families to keep, rest will be grouped into 'Other'

    for family in families[:cutoff]:
        if "Other" in family:
            modified_families["Other"] += family_members[family]
        # Rename all GPCRs
        elif "protein-coupled" in family:
            modified_families["G protein-coupled receptor"] += family_members[family]
        else:
            modified_families[family] = family_members[family]

    # All small families are now in 'Other' category
    for family in families[cutoff:]:
        modified_families["Other"] += family_members[family]

    #######
    # Create a dictionary of families that is ordered by the number of members
    families, family_members = zip(*modified_families.items())
    n_members = [len(v) for v in family_members]
    # Get sort order
    order = np.argsort(n_members)[::-1]
    n_members.sort()
    fams = list(families)
    families = [fams[i] for i in order]
    ordered_families = OrderedDict(zip(families, n_members[::-1]))

    return ordered_families
