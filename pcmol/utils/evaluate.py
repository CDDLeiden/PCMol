import numpy as np
import pandas as pd
import wandb
import os
import datetime

from pcmol.utils import PapyrusStandardizer
from pcmol.config import EvaluatorConfig

from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
from rdkit import DataStructs

from pcmol.utils.metrics import SAScore, calculate_property


def add_property(df, smiles, prop):
    """
    Add a property column to a dataframe
    """
    props = calculate_property(smiles, prop)
    df[prop] = props
    return df


def bulk_similarity(
    smiles: list, target_smiles: list, target_fps: list = None, metric: str = "tanimoto"
):
    """
    Calculates the similarity between a list of SMILES and a target list of SMILES
    using Bulk Tanimoto or Dice similarity
    Arguments:
        smiles (list)             : list of SMILES
        target (list)             : list of SMILES
        metric (str)              : similarity metric (tanimoto or dice)
    """

    if metric == "tanimoto":
        metric = DataStructs.TanimotoSimilarity
    elif metric == "dice":
        metric = DataStructs.DiceSimilarity
    else:
        raise ValueError("Invalid metric")

    # Convert SMILES to rdkit-molecules
    mols = [None for _ in smiles]
    fps = [None for _ in smiles]
    for i, smi in enumerate(smiles):
        if smi != "---":
            try:
                mol = mols[i] = Chem.MolFromSmiles(smi)
                fps[i] = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    mol, 2, nBits=2048
                )
            except:
                mols[i] = None
                fps[i] = None

    if target_fps is None:
        target_mols = [None for _ in target_smiles]
        for i, smi in enumerate(target_smiles):
            if smi != "---":
                try:
                    target_mols[i] = Chem.MolFromSmiles(smi)
                except:
                    target_mols[i] = None
        target_fps = [
            rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            for mol in target_mols
            if mol is not None
        ]

    # Calculate morgan fingerprints

    # Calculate similarity
    similarity = np.zeros((len(smiles), len(target_fps)))
    for i, fp in enumerate(fps):
        for j, target_fp in enumerate(target_fps):
            if fp is not None and target_fp is not None:
                try:
                    similarity[i, j] = metric(fp, target_fp)
                except:
                    try:
                        mol = Chem.MolFromSmiles(target_smiles[i])
                        target_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                            mol, 2, nBits=2048
                        )
                        similarity[i, j] = metric(fp, target_fp)
                    except:
                        similarity[i, j] = 0
    return similarity


class Evaluator:
    def __init__(self, model_runner, config: EvaluatorConfig, use_wandb: bool = False):
        self.config = config
        self.standardizer = PapyrusStandardizer()
        self.dataset = self.load_set(
            config.eval_dataset_path
        )  # Training set with fingerprints, index is canonical SMILES
        self.model_runner = model_runner
        self.eval_dir = os.path.join(self.model_runner.config.model_dir, "evals")
        self.use_wandb = use_wandb
        self.log = None
        self.eval_counter = 0

    def load_set(self, path):
        """
        Loads a dataset from a CSV file, converts SMILES to canonical SMILES
        and creates a hash column for faster lookup
        """
        # if not self.config.run_eval:
        return None
        print(f"Loading evaluation dataset from {path}...")
        df = pd.read_csv(path, sep="\t")
        ## Create hash column for faster lookup
        # print('Creating hash column by canonizing SMILES...')

        df.index = df.canonical
        ## Add train column

        train = [1 for _ in range(len(df))]
        df["train"] = train
        return df

    def valid(self, smiles):
        """
        Check if a SMILES string is valid
        """
        try:
            return self.standardizer(smiles)[0] is not None
        except:
            return False

    def to_canonical(self, smiles):
        """
        Convert a SMILES string to canonical SMILES
        """
        try:
            smiles = self.standardizer(smiles)[0]
            if smiles is None:
                return None
        except:
            return None
        return Chem.CanonSmiles(smiles)

    def evaluate(
        self,
        list_of_smiles: list,
        target: str,
        superset: list = None,
        calc_similarity: bool = False,
        metric: str = "tanimoto",
        calc_molprops: bool = False,
        train_mode=False,
    ):
        """
        Evaluate a list of SMILES strings

        Arguments:
            list_of_smiles (list)     : list of SMILES strings
            superset (list)           : list of SMILES strings to compare against (e.g. generated for other targets)
            calc_similarity (bool)    : whether to calculate similarity to a target
            metric (str)              : similarity metric (tanimoto or dice)
            calc_molprops (bool)      : whether to calculate molecule properties
            train_mode (bool)         : whether to save the evaluation results

        Returns:
            result_df (pd.DataFrame)  : dataframe with evaluation results
        """
        if train_mode:
            os.makedirs(self.eval_dir, exist_ok=True)

        print(f"    Evaluating {len(list_of_smiles)} SMILES strings...")
        result_df = pd.DataFrame(index=list_of_smiles)
        result_df["target"] = [target for _ in list_of_smiles]
        # result_df['SMILES'] = list_of_smiles
        dataset = self.dataset

        # assert target in dataset.target_id.unique(), 'Invalid target'

        if target in self.config.train_ids:
            result_df["train"] = [1 for _ in list_of_smiles]
        else:
            result_df["train"] = [0 for _ in list_of_smiles]

        ## Canonicalize
        legit = []
        for i, smi in enumerate(list_of_smiles):
            try:
                list_of_smiles[i] = self.to_canonical(smi)
            except:
                list_of_smiles[i] = "-----"

        result_df.index = list_of_smiles
        ## Validity
        result_df["valid"] = [self.valid(smi) for smi in list_of_smiles]

        ## Uniqueness
        # result_df['unique'] = [smi not in list_of_smiles for smi in list_of_smiles]
        if superset is not None:
            result_df["unique_overall"] = [
                smi not in superset for smi in list_of_smiles
            ]

        ## Molecule properties
        props = [
            "logP",
            "MW",
            "HBA",
            "HBD",
            "qed",
            "tpsa",
            "rotatable_bonds",
            "aromatic_rings",
            "num_rings",
            "heavy_atoms",
        ]
        if calc_molprops:
            for prop in props:
                result_df[prop] = calculate_property(list_of_smiles, prop)

        ## If the evaluation dataset is not available, skip the rest
        if self.config.run_eval:
            ## Novelty
            result_df["novel_overall"] = [
                not self.contained_in(smi, dataset) for smi in list_of_smiles
            ]
            if target is not None:
                result_df["novel"] = [
                    not self.contained_in(smi, dataset[dataset.target_id == target])
                    for smi in list_of_smiles
                ]

            # Similarity
            if calc_similarity:
                target_smiles = dataset[dataset.target_id == target].SMILES.tolist()
                target_fps = dataset[dataset.target_id == target].ECFP6.tolist()
                similarity = bulk_similarity(
                    list_of_smiles,
                    target_smiles=target_smiles,
                    target_fps=target_fps,
                    metric=metric,
                )

                result_df["similarity_mean"] = similarity.mean(axis=1)
                result_df["similarity_std"] = similarity.std(axis=1)
                result_df["similarity_max"] = similarity.max(axis=1)
                result_df["similarity_min"] = similarity.min(axis=1)

        return result_df

    def contained_in(self, smiles, df):
        """
        Check if a SMILES string is contained in a dataframe by doing a lookup in the index
        Index is canonical smiles
        (ugly but fast)
        """
        if self.config.canonize:
            try:
                _ = df.loc[smiles]
                return True
            except:
                return False
        else:
            return smiles in df.SMILES.tolist()

    def test_model(self, save=True, eval_index: int = 0):
        """
        Runs a testing routine:
        - Generates n molecules for each target from train_ids and test_ids
        - Evaluates the generated molecules
        - Returns a dictionary with the results
        """

        master_df = pd.DataFrame()

        for target in self.config.test_ids + self.config.train_ids:
            # if target not in self.dataset.target_id.unique():
            # continue
            print(f'Target: {target} {" "*100}')
            smiles, _ = self.model_runner.targetted_generation(
                target, repeat=self.config.budget, batch_size=self.config.batch_size
            )
            result_df = self.evaluate(
                smiles, target=target, calc_similarity=True, calc_molprops=True
            )
            master_df = pd.concat([master_df, result_df])

        if self.use_wandb:
            ## Calculate averages based on target_id
            meaned = master_df.groupby("target").mean()
            meaned["target"] = meaned.index
            meaned = meaned.reset_index(drop=True)
            meaned["eval_index"] = [self.eval_counter for _ in range(len(meaned))]
            if self.log is None:
                self.log = meaned
            else:
                self.log = pd.concat([self.log, meaned])
            wandb.log({"Evals": wandb.Table(dataframe=self.log)})

        if save:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            path = os.path.join(self.eval_dir, f"eval_{timestamp}.csv")
            master_df.to_csv(path, index=False)
        self.eval_counter += 1

        return master_df
