from dataclasses import dataclass
import yaml
import torch
import os

"""
This file contains the configuration for the model, dataset, trainer and evaluator
The RunnerConfig class contains all the other configs and can be used to save and load configs
"""


class dirs:
    """
    Contains the paths to the data directories
    """

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    PAPYRUS_DIR = os.path.join(DATA_DIR, "papyrus")
    MODEL_DIR = os.path.join(DATA_DIR, "models")
    ALPHAFOLD_DIR = os.path.join(DATA_DIR, "alphafold")
    RESULTS_DIR = os.path.join(DATA_DIR, "results")


###########
### Configs
@dataclass
class ModelConfig:
    d_emb: int = 384
    d_model: int = 512
    d_feedforward: int = 512
    n_heads: int = 32
    n_layers: int = 12
    dropout: float = 0.1
    pcm: bool = False  # Whether to use pchembl auxiliary loss
    loss_coefficients = dict(generative=1.0, pchembl=0.0)
    use_pooling = False
    pchembl_arch = [1024, 1024, 256, 1]


@dataclass
class TrainerConfig:
    epochs: int = 10
    batch_size: int = 96
    eval_every_n_batches: int = 1000
    save_every_n_batches: int = 100
    lr: float = 9e-5
    warmup_steps: int = 100
    devices = [1, 2, 3]
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    trained_for: int = 0  # Total number of samples seen
    parameter_count: int = 0  # Total number of parameters in the model


@dataclass
class DatasetConfig:
    dataset_dir: str = os.path.join(dirs.DATA_DIR, "training")
    dataset_prefix: str = "aug_10_no_orphans"  #'final_augmented'
    alphafold_dir: str = os.path.join(dirs.DATA_DIR, "alphafold")
    max_smiles_len: int = 102
    embedding_size: int = 1536
    embedding_type: str = "structure"


@dataclass
class EvaluatorConfig:
    """
    Configuration for the evaluator
    """

    eval_dataset_path: str = os.path.join(dirs.PAPYRUS_DIR, "filtered_w_features.tsv")
    train_ids = ["Q99685", "P29274", "P41597"]  # For tracking train set metrics
    test_ids = ["Q9BXC1", "Q13304", "Q5U431"]  # For tracking test set metrics
    wandb_logging: bool = True
    batch_size: int = 16
    budget: int = 5
    canonize: bool = (
        False  # Whether to canonize SMILES strings in the eval dataset before evaluation
    )
    run_eval: bool = False  # Whether to run evaluation after each training epoch


@dataclass
class RunnerConfig:
    """
    Main configuration for the runner
    Contains the model, dataset, trainer and evaluator configs
    Saves the config to a .yaml file
    Can load a config from a .yaml file
    """

    model: ModelConfig = ModelConfig()
    dataset: DatasetConfig = DatasetConfig()
    trainer: TrainerConfig = TrainerConfig()
    evaluator: EvaluatorConfig = EvaluatorConfig()
    model_dir: str = "trained_models"
    model_id: str = ""
    use_wandb: bool = False

    @staticmethod
    def load(config_file_path: str):
        """
        Loads a config from a .yaml file and returns a RunnerConfig object
        """
        ## Check if file exists
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"File {config_file_path} not found")

        with open(config_file_path, "r") as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)

        config = RunnerConfig(
            model=ModelConfig(**config_dict["model"]),
            dataset=DatasetConfig(**config_dict["dataset"]),
            trainer=TrainerConfig(**config_dict["trainer"]),
            evaluator=EvaluatorConfig(**config_dict["evaluator"]),
        )

        return config

    def save(self):
        """
        Saves the config of the runner to a .yaml file
        """
        config_dict = dict(
            model=self.model.__dict__,
            dataset=self.dataset.__dict__,
            trainer=self.trainer.__dict__,
            evaluator=self.evaluator.__dict__,
        )

        with open(f"{self.model_dir}/config.yaml", "w") as f:
            yaml.dump(config_dict, f)

        return config_dict
