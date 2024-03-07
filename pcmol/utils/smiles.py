import torch
import numpy as np
import re
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.MolStandardize import rdMolStandardize
from pcmol.utils.smiles_enumerator import SmilesEnumerator

opts = Draw.DrawingOptions()
Draw.SetComicMode(opts)


def generate_alternative_smiles(smiles, n_alternatives=10, n_attempts=500):
    """
    Generate alternative SMILES for a given SMILES string
    Parameters:
        smiles: SMILES string
        n_alternatives: number of alternative SMILES to generate
        n_attempts: number of attempts to generate up to n_alternative SMILES
    Returns:
        list of alternative SMILES
    """
    enumerator = SmilesEnumerator()
    alternatives, sdict = [], {}
    for i in range(n_attempts):
        alt_smiles = enumerator.randomize_smiles(smiles)
        # Hash to detect duplicates
        if sdict.get(alt_smiles) is None:
            sdict[alt_smiles] = 1
            alternatives.append(alt_smiles)
        if len(alternatives) == n_alternatives:
            break

    return alternatives


"""
All of the following functions are based on DrugEx codebase (https://github.com/CDDLeiden/DrugEx). 
"""


def clean_mol(smile, is_deep=True):
    """Taken from dataset.py, modified to take/return a single smile"""

    smile = (
        smile.replace("[O]", "O")
        .replace("[C]", "C")
        .replace("[N]", "N")
        .replace("[B]", "B")
        .replace("[2H]", "[H]")
        .replace("[3H]", "[H]")
    )
    try:
        mol = Chem.MolFromSmiles(smile)
        if is_deep:
            mol = rdMolStandardize.ChargeParent(mol)
        smileR = Chem.MolToSmiles(mol, 0)
        smile = Chem.CanonSmiles(smileR)
    except:
        print("Parsing Error:", smile)
        smile = None
    return smile


class VocSmiles:
    """
    A class for handling encoding/decoding from SMILES to an array of indices
    Taken from utils/vocab.py, slightly adjusted to fix bugs (token duplication)
    """

    def __init__(self, init_from_file=None, max_len=100):
        self.control = ["_", "GO"]
        from_file = []
        if init_from_file:
            from_file = self.init_from_file(init_from_file)
            from_file = list(set(from_file))
            from_file.sort()
        self.words = self.control + from_file
        self.size = len(self.words)
        self.tk2ix = dict(zip(self.words, range(len(self.words))))
        self.ix2tk = {v: k for k, v in self.tk2ix.items()}
        self.max_len = max_len

    def encode(self, input):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        output = torch.zeros(len(input), self.max_len).long()
        for i, seq in enumerate(input):
            for j, char in enumerate(seq):
                output[i, j] = self.tk2ix[char]
        return output

    def decode(self, tensor, is_tk=True):
        """Takes an array of indices and returns the corresponding SMILES"""
        tokens = []
        for token in tensor:
            if not is_tk:
                token = self.ix2tk[int(token)]
            if token == "EOS":
                break
            if token in self.control:
                continue
            tokens.append(token)
        smiles = "".join(tokens)
        smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smiles

    def split(self, smile):
        """Takes a SMILES and return a list of characters/tokens"""
        regex = "(\[[^\[\]]{1,6}\])"
        smile = smile.replace("Cl", "L").replace("Br", "R")
        tokens = []
        for word in re.split(regex, smile):
            if word == "" or word is None:
                continue
            if word.startswith("["):
                tokens.append(word)
            else:
                for i, char in enumerate(word):
                    tokens.append(char)
        return tokens + ["EOS"]

    def init_from_file(self, file):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        words = []
        with open(file, "r") as f:
            chars = f.read().split()
            words += sorted(set(chars))
        return words

    def calc_voc_fp(self, smiles, prefix=None):
        fps = np.zeros((len(smiles), self.max_len), dtype=np.long)
        for i, smile in enumerate(smiles):
            smile = clean_mol(smile)
            token = self.split(smile)
            if prefix is not None:
                token = [prefix] + token
            if len(token) > self.max_len:
                continue
            if {"C", "c"}.isdisjoint(token):
                continue
            if not {"[Na]", "[Zn]"}.isdisjoint(token):
                continue
            fps[i, :] = self.encode(token)
        return fps


def standardize_mol(mol):
    """
    Standardizes SMILES and removes fragments
    Arguments:
        mols (lst)                : list of rdkit-molecules
    Returns:
        smiles (set)              : set of SMILES
    """

    charger = rdMolStandardize.Uncharger()
    chooser = rdMolStandardize.LargestFragmentChooser()
    disconnector = rdMolStandardize.MetalDisconnector()
    normalizer = rdMolStandardize.Normalizer()
    carbon = Chem.MolFromSmarts("[#6]")
    salts = Chem.MolFromSmarts("[Na,Zn]")
    try:
        mol = disconnector.Disconnect(mol)
        mol = normalizer.normalize(mol)
        mol = chooser.choose(mol)
        mol = charger.uncharge(mol)
        mol = disconnector.Disconnect(mol)
        mol = normalizer.normalize(mol)
        smileR = Chem.MolToSmiles(mol, 0)
        # remove SMILES that do not contain carbon
        if len(mol.GetSubstructMatches(carbon)) == 0:
            return None
        # remove SMILES that still contain salts
        if len(mol.GetSubstructMatches(salts)) > 0:
            return None
        return Chem.CanonSmiles(smileR)
    except:
        print("Parsing Error:", Chem.MolToSmiles(mol))

    return None


def check_smiles(smiles, frags=None):
    shape = (len(smiles), 1) if frags is None else (len(smiles), 2)
    valids = np.zeros(shape)
    for j, smile in enumerate(smiles):
        # 1. Check if SMILES can be parsed by rdkit
        try:
            mol = Chem.MolFromSmiles(smile)
            valids[j, 0] = 0 if mol is None else 1
        except:
            valids[j, 0] = 0
    return valids


standard_grid = Chem.Draw.MolsToGridImage


def interactive_grid(mols, *args, molsPerRow=10, **kwargs):
    import mols2grid

    return mols2grid.display(mols, *args, n_cols=molsPerRow, **kwargs)


def smilesToGrid(smiles, *args, molsPerRow=10, **kwargs):
    mols = []
    for smile in smiles:
        try:
            m = Chem.MolFromSmiles(smile)
            if m:
                AllChem.Compute2DCoords(m)
                mols.append(m)
            else:
                raise Exception(f"Molecule empty for SMILES: {smile}")
        except Exception as exp:
            pass

    return interactive_grid(mols, *args, molsPerRow=molsPerRow, **kwargs)


def compare_gen_and_data(data_df, gen_df, target_ids, mols_per_row=5):
    """
    Generates a grid of molecules from the generator and the data
    To be used with the interactive grid
    Arguments:
        gen_df (pd.DataFrame)     : generator dataframe
        data_df (pd.DataFrame)    : data dataframe
        target_ids (str or list)  : target ids
        mols_per_row (int)        : number of molecules per row
    Returns:
        smiles (list)             : list of SMILES (row1: data, row2: gen)
    """
    if type(target_ids) == str:
        target_ids = [target_ids]

    gen_smiles, data_smiles = [], []
    for target_id in target_ids:
        gen_smiles += gen_df[gen_df["target_id"] == target_id]["SMILES"].tolist()
        data_smiles += data_df[data_df["target_id"] == target_id]["SMILES"].tolist()

    gen_smiles = list(np.random.choice(gen_smiles, size=mols_per_row, replace=False))
    data_smiles = list(np.random.choice(data_smiles, size=mols_per_row, replace=False))

    return gen_smiles + data_smiles
