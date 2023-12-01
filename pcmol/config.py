from dataclasses import dataclass
import yaml
import torch
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PAPYRUS_DIR = os.path.join(DATA_DIR, 'papyrus')
MODEL_DIR = os.path.join(DATA_DIR, 'models')

### Configs
@dataclass
class ModelConfig:
    d_emb:         int = 384
    d_model:       int = 512
    d_feedforward: int = 512
    n_heads:       int = 32
    n_layers:      int = 12
    dropout:       float = 0.1
    pcm:           bool = False # Whether to use pchembl auxiliary loss
    loss_coefficients = dict(
        generative=1.0,
        pchembl=0.0)
    use_pooling = False
    pchembl_arch = [1024, 1024, 256, 1]


@dataclass
class TrainerConfig:
    epochs:               int = 10
    batch_size:           int = 96
    eval_every_n_batches: int = 1000
    save_every_n_batches: int = 100
    lr:                   float = 9e-5
    warmup_steps:         int = 100  
    devices =             [1, 2, 3]
    dev =                 torch.device('cuda') 
    trained_for:          int = 0 # Total number of samples seen
    parameter_count:      int = 0 # Total number of parameters in the model


@dataclass
class DatasetConfig:
    dataset_dir   : str = os.path.join(DATA_DIR, 'training')
    dataset_prefix: str = 'aug_10_no_orphans' #'final_augmented'
    alphafold_dir:  str = os.path.join(DATA_DIR, 'alphafold')
    max_smiles_len: int = 102
    embedding_size: int = 1536
    embedding_type: str = 'structure'


@dataclass
class EvaluatorConfig:
    """
    Configuration for the evaluator
    """
    run_eval: bool = False
    dataset_path: str = os.path.join(PAPYRUS_DIR, 'filtered_w_features.tsv')
    train_ids = ['Q99685', 'P29274', 'P41597'] # For tracking train set metrics
    test_ids = ['Q9BXC1', 'Q13304', 'Q5U431']  # For tracking test set metrics
    wandb_logging: bool = True
    batch_size: int = 16
    budget: int = 5
    canonize: bool = False # Whether to canonize SMILES strings in the eval dataset before evaluation


@dataclass
class RunnerConfig:
    """ 
    Main configuration for the runner
    """
    model:   ModelConfig   = ModelConfig()
    dataset: DatasetConfig = DatasetConfig()
    trainer: TrainerConfig = TrainerConfig()
    evaluator: EvaluatorConfig = EvaluatorConfig()
    model_dir: str = MODEL_DIR
    model_id:  str = ''
    use_wandb: bool = False


def load_config(config_file_path: str):
    """
    Loads a config from a .yaml file and returns a RunnerConfig object
    """
    with open(config_file_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    config = RunnerConfig(
        model=ModelConfig(**config_dict['model']),
        dataset=DatasetConfig(**config_dict['dataset']),
        trainer=TrainerConfig(**config_dict['trainer']),
        evaluator=EvaluatorConfig(**config_dict['evaluator']))
    
    return config

def save_config(runner):
    """
    Saves the config of the runner to a .yaml file
    """
    config_dict = dict(
        model   = runner.config.model.__dict__,
        dataset = runner.config.dataset.__dict__,
        trainer = runner.config.trainer.__dict__,
        evaluator = runner.config.evaluator.__dict__)

    with open(f'{runner.config.model_dir}/config.yaml', 'w') as f:
        yaml.dump(config_dict, f)

    return config_dict
