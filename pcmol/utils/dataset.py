import os
import shutil
import torch
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pcmol.utils.smiles import VocSmiles
from torch.utils.data import Dataset as TorchDataset
from pcmol.config import DatasetConfig
from pcmol.utils.smiles import generate_alternative_smiles
import matplotlib.pyplot as plt
from pcmol.config import dirs


## Voc and dataset
def load_voc(dataset_config: DatasetConfig):
    """
    Loads the vocabulary from config (default: DatasetConfig initialized from config.py)
    """
    dataset_prefix = dataset_config.dataset_prefix
    dataset_dir = dataset_config.dataset_dir
    voc_file = os.path.join(dataset_dir, f"{dataset_prefix}", "voc_smiles.txt")
    return VocSmiles(voc_file, max_len=dataset_config.max_smiles_len)


def load_dataset(config: DatasetConfig = None, pre_load=False, load_all=True):
    """
    Loads the dataset from config (default: DatasetConfig initialized from config.py)
    """
    # Load base config if none is provided
    if config is None:
        config = DatasetConfig()
    dataset_prefix = config.dataset_prefix
    alphafold_embedding_dir = config.alphafold_dir
    embedding_type = config.embedding_type
    voc_smiles = load_voc(config)
    dataset_dir = os.path.join(config.dataset_dir, f"{dataset_prefix}")
    if load_all:
        protein_list = None
    else:
        protein_list = os.path.join(dataset_dir, f"{dataset_prefix}.txt")

    dataset_file = os.path.join(dataset_dir, "train.tsv")

    dataset = ProteinSmilesDataset(
        alphafold_embedding_dir,
        dataset_file=dataset_file,
        voc_smiles=voc_smiles,
        protein_set=protein_list,
        embedding_type=embedding_type,
        pre_load=pre_load,
    )
    return dataset


class ProteinDataset:
    """
    Class for loading and storing AlphaFold embeddings in torch.Tensor format

    Args:
        alphafold_embedding_dir (str): Path to the directory containing the AlphaFold embeddings
        protein_set (str): Path to a file containing a list of protein ids to load (optional)
        embedding_type (str): Type of embedding to load (default: 'single', options: 'single', 'structure')
        embedding_size (int): Size of the embedding (default: 1024)
        pre_load (bool): Whether to pre-load all embeddings into memory (default: False)
    """

    def __init__(
        self,
        alphafold_embedding_dir,
        embedding_type="single",
        embedding_size=1536,
        protein_set=None,
        pre_load=False,
    ):
        self.alphafold_embedding_dir = alphafold_embedding_dir
        self.max_len = embedding_size
        self.lengths = {}
        self.emb_max = None
        self.emb_min = None

        # Embedding dict accessible via protein_id
        if pre_load:
            self.embeddings = self.load_protein_embeddings(
                protein_set=protein_set, embedding_type=embedding_type
            )
        else:
            self.embeddings = {}

    def get_protein_embedding(self, protein_id, embedding_type="structure"):
        """
        Returns a torch tensor of the target protein

        Args:
            protein_id (str): Protein id
            embedding_type (str): Type of embedding to load (default: 'single', options: 'single', 'structure')

        Returns:
            embedding (torch.Tensor): Protein embedding
            shape (tuple): Shape of the embedding
        """
        protein_dir = os.path.join(self.alphafold_embedding_dir, protein_id)
        embedding_file = os.path.join(protein_dir, f"{embedding_type}.npy")

        np_file = np.load(embedding_file)
        embedding = torch.from_numpy(np_file)
        n, f = embedding.shape
        if n >= self.max_len:
            return None, 0
        output = torch.zeros(self.max_len, f)
        output[:n, :] = embedding
        self.lengths[protein_id] = n
        return output, embedding.shape

    def load_protein_embeddings(
        self, protein_set=None, embedding_type="structure", scale=True, recalc=False
    ):
        """
        Loads all protein embeddings from the dataset directory

        Args:
            protein_set (str): Path to a file containing a list of protein ids to load (optional)
            embedding_type (str): Type of embedding to load (default: 'single', options: 'single', 'structure')
            scale (bool): Whether to scale the embeddings in the range [-1, 1] (default: True)
            recalc (bool): Whether to recalculate the min/max values for scaling (default: False)

        Returns:
            protein_dict (dict): Dictionary of protein embeddings accessible via protein_id
        """
        protein_dict = {}
        protein_ids = [
            folder
            for folder in os.listdir(self.alphafold_embedding_dir)
            if os.path.isdir(os.path.join(self.alphafold_embedding_dir, folder))
        ]

        if protein_set is not None:
            with open(protein_set, "r") as pids:
                pids = pids.readlines()
                pids = [p.strip("\n") for p in pids]
            protein_ids = list(set(pids).intersection(set(protein_ids)))
        protein_ids.sort()

        # Scaling in the range [-1, 1] (based on min/max values of each feature)
        embs, filtered_pids, lengths = [], [], []
        print(f"Loading AlphaFold2 <{embedding_type}> embeddings...")
        for pid in tqdm(protein_ids):
            emb, shape = self.get_protein_embedding(pid, embedding_type)
            if emb is not None:
                embs.append(emb)
                filtered_pids.append(pid)
                lengths.append(shape[0])

        concat = torch.cat(embs, dim=0)
        # Store embedding min/max values for scaling test set embeddings
        if recalc:
            emb_max = self.emb_max = torch.max(concat, dim=0)[0]
            emb_min = self.emb_min = torch.min(concat, dim=0)[0]
        else:
            emb_min = np.load(os.path.join(dirs.ALPHAFOLD_DIR, "emb_min.npy"))
            emb_max = np.load(os.path.join(dirs.ALPHAFOLD_DIR, "emb_max.npy"))
            self.emb_min = emb_min
            self.emb_max = emb_max

        ## Scale embeddings
        if scale:
            for i, pid in enumerate(filtered_pids):
                scaled_embedding = (embs[i] - emb_min) / (emb_max - emb_min) * 2 - 1
                # Pad with zeros
                scaled_embedding[lengths[i] :, :] = 0
                protein_dict[pid] = scaled_embedding

        return protein_dict


class ProteinSmilesDataset(TorchDataset):
    """
    Class for fetching both AF embeddings and Papyrus SMILES data

    Args:
        alphafold_embedding_dir (str): Path to the directory containing the AlphaFold embeddings
        dataset_file (str): Path to the dataset file
        voc_smiles (VocSmiles): Vocabulary object
        pre_encode_smiles (bool): Whether to pre-tokenize the SMILES strings (default: False)
            *** Useful for speeding up training (pre-computing the SMILES tokens)
        pre_load (bool): Whether to pre-load all embeddings into memory (default: False)
        protein_set (str): Path to a file containing a list of protein ids to load (optional)
    """

    def __init__(
        self,
        alphafold_embedding_dir: str,
        dataset_file: str,
        voc_smiles: VocSmiles,
        pre_load=False,
        protein_set=None,
        train=False,
        **kwargs,
    ) -> None:

        self.proteins = ProteinDataset(
            alphafold_embedding_dir,
            pre_load=pre_load,
            protein_set=protein_set,
            **kwargs,
        )
        self.voc_smiles = voc_smiles
        if train:
            self.tsv_dataset = self.read_dataset(dataset_file)
            print(f"Dataset: len: {len(self)}")
            self.dataframe = read_dataset(dataset_file)
        else:
            self.tsv_dataset = None
        self.encoded_smiles = []
        self.error_count = 0

        # When training this should be set to True (when training for multiple epochs)
        self.pre_load = pre_load
        # if pre_load:
        #     self.encode_smiles()

    def encode_smiles(self):
        """
        Encodes a SMILES string into a list of integers
        Should be used before training to save processing time
        """
        self.encoded_smiles = []
        for i in tqdm(range(len(self.tsv_dataset))):
            _, _, smiles = self.tsv_dataset[i].split("\t")
            smiles = smiles.strip("\n").split(" ")
            self.encoded_smiles += [self.voc_smiles.encode([smiles])]

    def __len__(self):
        return len(self.tsv_dataset)

    def __getitem__(self, idx):
        try:
            pid, pchembl, smiles = self.tsv_dataset[idx].split("\t")
            pchembl = torch.tensor(float(pchembl))
            smiles = smiles.strip("\n").split(" ")
            protein_embedding = self.proteins.embeddings[pid]
            encoded_smiles = self.voc_smiles.encode([smiles])
        except Exception as e:
            self.error_count += 1
            print("Error in __getitem__", e, idx, self.error_count)
            return self.__getitem__(idx + random.randint(1, 10))

        return protein_embedding, encoded_smiles, pchembl

    def write_dataset(self, dataset_path):
        """
        Writes the dataset to a file
        """
        with open(dataset_path, "w") as data:
            for line in self.tsv_dataset:
                data.write(line)

    def read_dataset(self, dataset_path):
        """
        Reads the dataset from a file
        """
        print(f"Reading SMILES dataset from {dataset_path}...")
        with open(dataset_path, "r") as data:
            data = data.readlines()[1:]
            return data

    def get_molecules(self, target_id):
        """
        Returns a list of SMILES strings for a given target
        """
        smiles = self.dataframe[self.dataframe["target_id"] == target_id][
            "SMILES"
        ].tolist()
        return smiles

    def embedding_available(self, target_id):
        """
        Checks if the protein embedding is available
        """
        return target_id in self.proteins.embeddings


def clean_dataset(af_output_dir, output_dir):
    """
    Processes the raw AlphaFold output and saves it in a more convenient format
    """
    protein_ids = [
        folder
        for folder in os.listdir(af_output_dir)
        if os.path.isdir(os.path.join(af_output_dir, folder))
    ]
    print(protein_ids)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "proteins"), exist_ok=True)

    # Process AlphaFold's output
    for protein in protein_ids:

        src_dir = os.path.join(af_output_dir, protein)
        protein_dir = os.path.join(output_dir, "proteins", protein)
        os.makedirs(protein_dir, exist_ok=True)

        path = os.path.join(src_dir, "result_model_1_pred_0.pkl")
        pkl = open(path, "rb")
        result = pickle.load(pkl)
        reps = result["representations"]
        single, struct = reps["single"], reps["structure_module"]

        np.save(os.path.join(protein_dir, "single.npy"), single)
        np.save(os.path.join(protein_dir, "struct.npy"), struct)

        files_to_copy = [
            "ranked_0.pdb",
            "unrelaxed_model_1_pred_0.pdb",
            f"{protein}.fasta",
        ]

        for file in files_to_copy:
            try:
                shutil.copy(
                    os.path.join(src_dir, file), os.path.join(protein_dir, file)
                )
            except:
                pass


class Ligand:
    def __init__(self, string, target, pchembl) -> None:
        self.string = string
        self.length = len(string)
        self.target = target
        self.pchembl = pchembl

    def __len__(self) -> int:
        return len(self.string)

    def __repr__(self) -> str:
        return f"{self.target} L={len(self)} | {self.pchembl}"


def read_dataset(path):
    """
    Reads the dataset and returns a list of strings with the following format
        protein_id \t SMILES \t pchembl
    """
    dataset = pd.read_csv(path, sep="\t")
    return dataset


def augment_ligands(ligand_list, num_augmentations, weighted_sampling=True, r=1.5):
    """
    Augments the ligands in the ligand_list by sampling from the list of ligands
    generate_alternative_smiles(ligand, n_alternatives) is used to generate the alternative smiles
    via enumeration

    Args:
        ligand_list (list): List of Ligand objects
        num_augmentations (int): Number of augmentations to generate
        weighted_sampling (bool): Whether to sample the ligands from a weighted distribution
            based on the pchembl values
        r (float): Exponent for the pchembl weights (default: 1.5)
    """
    augmented = []

    # Sampling weights
    # base coefficients for pchembl values:
    #   pchembl=6.5 -> 1
    #   pchembl=7.5 -> 2
    #   pchembl=8.5 -> 3
    # ^r introduces a nonlinear bias towards ligands with higher pchembl values
    # e.g. a ligand with pchembl=8.5 will have a weight of 3^r = 5.2, while a ligand with pchembl=6.5
    # will have a weight of 1^r = 1, i.e. the ligand with pchembl=8.5 will be sampled 5.2 times more often

    if weighted_sampling:
        # Sample smiles to be augmented from the pchembl-weighted distribution
        sampling_weights = [(ligand.pchembl - 5.5) ** r for ligand in ligand_list]
        sampled = random.choices(
            ligand_list, k=num_augmentations, weights=sampling_weights
        )
    else:
        # Sample smiles to be augmented uniformly
        sampled = random.choices(ligand_list, k=num_augmentations)

    for i, ligand in enumerate(ligand_list):
        n_alternatives = sampled.count(ligand)
        alt_smiles = generate_alternative_smiles(
            ligand.string,
            n_alternatives=n_alternatives,
            n_attempts=int(n_alternatives * 4),
        )
        augmented += [
            Ligand(smiles, ligand.target, ligand.pchembl) for smiles in alt_smiles
        ]
    return augmented


def create_dataset(df, pids, min_num_smiles, max_num_smiles, coeff_dict):
    """
    Creates a dataset with a minimum number of smiles per protein

    Args:
        df (pd.DataFrame): Dataframe containing the dataset
        pids (list): List of proteins to augment
        min_num_smiles (int): Minimum number of smiles per protein
        max_num_smiles (int): Maximum number of smiles per protein
        coeff_dict (dict): Dictionary of augmentation coefficients for the number of smiles per protein
            Each is in the range [0, 1] and is used to scale the number of total augmentations, where
            0 -> min_num_smiles, 1 -> max_num_smiles
    """
    augmented = {}

    ## Iterate over proteins
    for j, target in enumerate(pids):
        smiles = df[df["target_id"] == target]["SMILES"].to_list()
        pchembl = df[df["target_id"] == target]["pchembl_value_Mean"].to_list()
        ligands = [
            Ligand(smiles_string, target, pchembl[i])
            for i, smiles_string in enumerate(smiles)
        ]
        num_smiles = len(smiles)

        ## Determine the number of augmentations
        num_augmentations = int(
            min_num_smiles - num_smiles + int(coeff_dict[target] * max_num_smiles)
        )
        if num_augmentations < min_num_smiles:
            num_augmentations = int(min_num_smiles)

        ## Augment the ligands
        ligands += augment_ligands(ligands, num_augmentations)
        print(
            f"{j:5} | Target: {target} | before aug: {len(smiles):8} | num_augmentations: {num_augmentations:5} | result: {len(ligands):5}",
            end="\r",
        )
        augmented[target] = ligands
    return augmented


def get_coefficients(dataframe, column="target_id", scaling_factor=1.1, plot=True):
    """
    Calculates the coefficients for the number of smiles per protein post augmentation
    Coefficients are in the range [0, 1] and are used to scale the number of total augmentations, where
    0 -> min_num_smiles, 1 -> max_num_smiles

    Since the number of smiles per protein is highly skewed and follows, a power law distribution,
    the coefficients are calculated as follows:
    - Performs a log transformation on the number of smiles per protein which results in a (close to) linear distribution
    - The values are then normalized in the range [0, 1]
    - The coefficients are then calculated as the normalized values multiplied by a scaling factor (e.g. if the scaling factor is 2.0,
      the coefficients will be in the range [0, 2.0])

    Args:
        dataframe (pd.DataFrame): Dataframe containing the dataset
        column (str): Column to group by (default: 'target_id')
        scaling_factor (float): Scaling factor for the coefficients (default: 1.1)
        plot (bool): Whether to plot the coefficients (default: True)

    Returns:
        coeff_dict (dict): Dictionary of coefficients accessible via protein_id
    """
    counts = dataframe["target_id"].value_counts().to_list()
    order = np.argsort(counts)
    n_smiles = np.array(counts)[order]
    pids = dataframe["target_id"].value_counts().index.to_list()
    pids = [pids[i] for i in order]
    logd = np.log(n_smiles)

    # normalize in range 0-1
    coefficients = (logd - logd.min()) / (logd.max() - logd.min()) * scaling_factor
    coeff_dict = dict(zip(pids, coefficients))

    if plot:
        plt.title("Coefficients for the number of smiles per protein")
        plt.bar(range(len(coefficients)), coefficients[::-1])
        plt.xlabel("Protein #")
        plt.ylabel("Coefficient")
        plt.show()

    return coeff_dict
