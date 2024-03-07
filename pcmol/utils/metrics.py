#
# calculation of synthetic accessibility score as described in:
#
# Estimation of Synthetic Accessibility Score of Drug-like Molecules based on Molecular Complexity and Fragment Contributions
# Peter Ertl and Ansgar Schuffenhauer
# Journal of Cheminformatics 1:8 (2009)
# http://www.jcheminf.com/content/1/1/8
#
# several small modifications to the original paper are included
# particularly slightly different formula for marocyclic penalty
# and taking into account also molecule symmetry (fingerprint density)
#
# for a set of 10k diverse molecules the agreement between the original method
# as implemented in PipelinePilot and this implementation is r2 = 0.97
#
# peter ertl & greg landrum, september 2013
#

from pcmol.utils.parallel import parallel_apply
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcNumHBA
from rdkit.Chem.rdMolDescriptors import CalcNumHBD
from rdkit.Chem.QED import qed
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.Chem.rdMolDescriptors import CalcNumAromaticRings
from rdkit.Chem.rdMolDescriptors import CalcNumRings
from rdkit.Chem.rdMolDescriptors import CalcNumHeavyAtoms

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pickle

import math
import os.path as op

_fscores = None


def readFragmentScores(name="fpscores"):
    import gzip

    global _fscores
    # generate the full path filename:
    if name == "fpscores":
        name = op.join(op.dirname(__file__), name)
    data = pickle.load(gzip.open("%s.pkl.gz" % name))
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict


def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro


def SAScore(m):
    if _fscores is None:
        readFragmentScores()

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(
        m, 2
    )  # <- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.0
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.0
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = (
        0.0
        - sizePenalty
        - stereoPenalty
        - spiroPenalty
        - bridgePenalty
        - macrocyclePenalty
    )

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.0
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * 0.5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11.0 - (sascore - min + 1) / (max - min) * 9.0
    # smooth the 10-end
    if sascore > 8.0:
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
    if sascore > 10.0:
        sascore = 10.0
    elif sascore < 1.0:
        sascore = 1.0

    return sascore


def calculate_property(smiles, prop):
    """
    Calculate a descriptor for a SMILES string
    """
    props_dict = dict(
        logP=MolLogP,
        MW=rdMolDescriptors.CalcExactMolWt,
        HBA=CalcNumHBA,
        HBD=CalcNumHBD,
        qed=qed,
        tpsa=CalcTPSA,
        rotatable_bonds=CalcNumRotatableBonds,
        aromatic_rings=CalcNumAromaticRings,
        num_rings=CalcNumRings,
        heavy_atoms=CalcNumHeavyAtoms,
        sa=SAScore,
    )

    molprop = props_dict[prop]
    descs = [None for _ in smiles]
    for i, smi in enumerate(smiles):
        if smi != "---":
            try:
                mol = Chem.MolFromSmiles(smi)
                descs[i] = molprop(mol)
            except:
                descs[i] = None
    return descs


### Functions for parallel apply (to avoid passing kwargs)
def SA(smiles):
    return calculate_property(smiles, "sa")


def logP(smiles):
    return calculate_property(smiles, "logP")


def MW(smiles):
    return calculate_property(smiles, "MW")


def HBA(smiles):
    return calculate_property(smiles, "HBA")


def HBD(smiles):
    return calculate_property(smiles, "HBD")


def qed(smiles):
    return calculate_property(smiles, "qed")


def tpsa(smiles):
    return calculate_property(smiles, "tpsa")


def rotatable_bonds(smiles):
    return calculate_property(smiles, "rotatable_bonds")


def aromatic_rings(smiles):
    return calculate_property(smiles, "aromatic_rings")


def num_rings(smiles):
    return calculate_property(smiles, "num_rings")


def heavy_atoms(smiles):
    return calculate_property(smiles, "heavy_atoms")


def add_mol_properties(df, column="SMILES", parallel=False, n_jobs=64, chunksize=1024):
    """
    Add molecular properties to a dataframe
    """
    fns = [
        logP,
        MW,
        HBA,
        HBD,
        qed,
        tpsa,
        rotatable_bonds,
        aromatic_rings,
        num_rings,
        heavy_atoms,
    ]

    for prop in fns:
        print(prop)
        if parallel:
            df[prop.__name__] = parallel_apply(
                prop, df, column=column, n_jobs=n_jobs, chunk_size=chunksize
            )
        else:
            df[prop.__name__] = df[column].apply(prop)

    return df
