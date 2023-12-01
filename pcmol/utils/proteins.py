
from urllib import request
from tqdm import tqdm
import os, sys
from collections import OrderedDict
import numpy as np
import csv

class ProteinSequence:
    """
    Simple object for storing protein info
    """
    def __init__(self, sequence: str=None, protein_id: str=None, filename: str=None, info: str=None):
        self.info = info
        self.seq = sequence
        self.protein_id = protein_id
        self.len = None
        self.ligand_idxs = [] # Datapoint line idxs in papyrus
        self.n_ligands = 0  # Number of datapoints in papyrus
    
    def __len__(self):
        return len(self.seq)

    def __repr__(self):
        return f"Protein Seq: {self.protein_id}, length: \
                {len(self)}, ligands: {self.n_ligands}"


def get_names(target_proteins):
    """
    Returns a dictionary of protein names for the target proteins (from Fasta headers)
    """
    folder = '/home/andrius/datasets/foldedPapyrus/proteins'
    proteins = [load_protein_sequence(pid, folder) for pid in target_proteins]
    headers = [p.info for p in proteins]
    name_strings = [h.split('|')[2].split('OS')[0] for h in headers]
    tags = [n.split(' ')[0] for n in name_strings]
    names = [' '.join([n.split(' ')[1:]]) for n in name_strings]
    target_proteins.sort()
    tag_dict  = OrderedDict(dict(zip(target_proteins, tags)))
    name_dict = OrderedDict(dict(zip(target_proteins, names)))
    
    return name_dict, tag_dict

def download_fasta(protein_id, output_folder):
    """
    Retrieves the fasta file from Uniprot based on protein ID
    """
    try:
        remote_url = f'https://uniprot.org/uniprot/{protein_id}.fasta'
        local_file = f'{output_folder}/{protein_id}.fasta'
        open(local_file, 'a').close()
        request.urlretrieve(remote_url, local_file)
    except:
        print(f'Fasta file for {protein_id} could not be downloaded.')

def read_fasta(file_path):
    """
    Reads a .fasta file and returns a Protein Sequence object
    """
    try:

        with open(file_path, 'r') as fasta:
            content = fasta.readlines()
            header = content[0]
            sequence = content[1:]
            stripped = [line.strip('\n') for line in sequence]
            sequence = ''.join(stripped)
            protein_id = header.split('|')[1].strip()
        
        protein_seq = ProteinSequence(sequence=sequence, protein_id=protein_id, info=header)
        return protein_seq
    except:
        return None

def load_protein_sequence(protein_id, data_folder) -> ProteinSequence:
    """
    If needed download the .fasta file based on protein ID and return a 
    ProteinSequence object
    """

    # Check if downloaded
    filename = f'{data_folder}/{protein_id}/{protein_id}.fasta'
    if not os.path.isfile(filename):
        download_fasta(protein_id, data_folder)
        
    protein_seq = read_fasta(filename)
    return protein_seq

def get_papyrus_proteins(papyrus_file, output_folder, start=0, end=61085165):
    """
    Scans through the whole papyrus dataset, and downloads all required .fasta files from 
    uniprot, based on protein accesion IDs (attribute[9] in papyrus)

    Returns a dict of ProteinSequence objects indexed via protein_ids
    """
    if not os.path.isfile(papyrus_file):
        print('Papyrus file not found.')
        sys.exit()

    end = 61085165 if not end else end
    
    with open(papyrus_file, 'r') as papyrus:

        header = papyrus.readline()
        proteins = {}

        for idx in tqdm(range(start, end)):
            entry = papyrus.readline()
            attributes = entry.split('\t')
            protein_id = attributes[9]

            if not proteins.get(protein_id):
                p_sequence = load_protein_sequence(protein_id, data_folder=output_folder)
                proteins[protein_id] = p_sequence
            else:
                proteins[protein_id].n_ligands += 1
    return proteins

def get_proteins(pids, output_folder):
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


def get_families(papyrus_targets_file: str, targets: list):

    with open(papyrus_targets_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        families = []
        target_families = {}
        for line in reader:
            target = line[0].strip('_WT')
            classes = line[5].split('->')
            family = classes[1] if len(classes) > 1 else classes[0]
            if family == '':
                family = 'Unknown'
            families += [family]
            target_families[target] = family

    families = list(set(families))
    ## Create a dictionary of families and their members

    family_members = {}
    for protein in targets:
        family = target_families[protein]
        if family_members.get(family) is None:
            family_members[family] = [protein]
        else:
            family_members[family] += [protein]

    ## Rename some families
    modified_families = OrderedDict()
    modified_families['Other'] = []
    modified_families['G protein-coupled receptor'] = []
    cutoff = 25 # Number of families to keep, rest will be grouped into 'Other'

    for family in families[:cutoff]:
        if 'Other' in family:
            modified_families['Other'] += family_members[family]
        # Rename all GPCRs
        elif 'protein-coupled' in family:
            modified_families['G protein-coupled receptor'] += family_members[family]
        else:
            modified_families[family] = family_members[family]

    # All small families are now in 'Other' category
    for family in families[cutoff:]:
        modified_families['Other'] += family_members[family]

    #######
    # Create a dictionary of families that is ordered by the number of members
    families, family_members = zip(*modified_families.items())
    n_members = [len(v) for v in family_members]
    # Get sort order
    order = np.argsort(n_members)[::-1]
    n_members.sort()
    fams = list(families)
    families = [fams[i] for i in order]
    targets = [family_members[i] for i in order]
    ordered_families = OrderedDict(zip(families, n_members[::-1]))
    families = list(family_members.keys())

    return ordered_families

def parse_family_data(papyrus_targets_file: str):

    with open(papyrus_targets_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        families = []
        target_families = {}
        for line in reader:
            target = line[0].strip('_WT')
            classes = line[5].split('->')
            family = classes[1] if len(classes) > 1 else classes[0]
            if family == '':
                family = 'Unknown'
            families += [family]
            target_families[target] = family

    families = list(set(families))
    ## Create a dictionary of families and their members

    family_members = {}
    for protein in targets:
        family = target_families[protein]
        if family_members.get(family) is None:
            family_members[family] = [protein]
        else:
            family_members[family] += [protein]

    ## Rename some families
    modified_families = OrderedDict()
    modified_families['Other'] = []
    modified_families['G protein-coupled receptor'] = []
    cutoff = 25 # Number of families to keep, rest will be grouped into 'Other'

    for family in families[:cutoff]:
        if 'Other' in family:
            modified_families['Other'] += family_members[family]
        # Rename all GPCRs
        elif 'protein-coupled' in family:
            modified_families['G protein-coupled receptor'] += family_members[family]
        else:
            modified_families[family] = family_members[family]

    # All small families are now in 'Other' category
    for family in families[cutoff:]:
        modified_families['Other'] += family_members[family]

    #######
    # Create a dictionary of families that is ordered by the number of members
    families, family_members = zip(*modified_families.items())
    n_members = [len(v) for v in family_members]
    # Get sort order
    order = np.argsort(n_members)[::-1]
    n_members.sort()
    fams = list(families)
    families = [fams[i] for i in order]
    targets = [family_members[i] for i in order]
    ordered_families = OrderedDict(zip(families, n_members[::-1]))
    families = list(family_members.keys())

    return ordered_families