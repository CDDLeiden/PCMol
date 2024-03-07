import torch
import wandb
import os
import time
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

from pcmol.config import RunnerConfig, dirs
from pcmol.models import AF2SmilesTransformer, count_parameters
from pcmol.utils.smiles import check_smiles
from pcmol.utils.dataset import load_dataset, load_voc
from pcmol.utils.evaluate import Evaluator
from pcmol.utils.downloader import download_protein_data

## Suppress RDKit warnings
from rdkit import RDLogger

rdlog = RDLogger.logger()
rdlog.setLevel(RDLogger.CRITICAL)

## Suppress torch warnings
import warnings

warnings.filterwarnings("ignore")


class Runner:
    """
    Main class for training and evaluating models

    Args:
        config: RunnerConfig                   - configuration for the runner
        model_id: str                          - ID of the model to load
        checkpoint: int                        - checkpoint number to load
        load: bool                             - whether to load the model weights
        device: str                            - device to run on, cuda or cpu
    """

    def __init__(
        self,
        config: RunnerConfig = None,
        model_id: str = None,
        checkpoint: int = 0,
        load_weights: bool = False,
        inference=False,
        device: str = "cuda",
    ) -> None:

        if model_id is not None:
            # Try to load the model
            model_dir = os.path.join(dirs.MODEL_DIR, str(model_id))
            print(f"Loading model from {model_dir}")
            if os.path.exists(model_dir):
                load_weights = True
                config_path = os.path.join(model_dir, "config.yaml")
                print("Loading config from", config_path)
                config = self.config = RunnerConfig.load(config_path)
                config.model_dir = model_dir

        if model_id is None and config is None:
            self.config = RunnerConfig()

        self.config = config
        self.config.trainer.dev = torch.device(device)

        # Model ID
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model_id = self.config.model_id = str(model_id) + str(self.timestamp)
        self.parent_model = model_id
        self.model_dir = os.path.join(dirs.MODEL_DIR, self.model_id)
        self.config.model_dir = self.model_dir
        self.config.model_id = self.model_id

        self.voc = load_voc(config.dataset)
        if not inference:
            self.dataset = load_dataset(config.dataset, pre_load=False)
        else:
            self.dataset = None
        self.model = AF2SmilesTransformer(
            self.voc, **config.model.__dict__, dev=config.trainer.dev
        )
        self.evaluator = Evaluator(self, config.evaluator, use_wandb=config.use_wandb)
        self.checkpoint = checkpoint

        if load_weights:
            ## Load the model weights
            weights = os.path.join(
                dirs.MODEL_DIR, model_id, f"model_{self.checkpoint}.pkg"
            )
            state_dict = torch.load(weights, map_location=self.config.trainer.dev)
            self.model.load_state_dict(state_dict)

        self.optim = torch.optim.AdamW(
            self.model.parameters(), lr=config.trainer.lr, betas=(0.95, 0.98)
        )

        self.scaler = torch.cuda.amp.GradScaler()
        param_count = count_parameters(self.model)
        self.config.trainer.parameter_count = param_count

        print(
            f"Loaded model {model_id}, parameter count: {param_count} \
              \nModel directory: {config.model_dir}, checkpoint: {checkpoint}"
        )

    def save_model(self, checkpoint: bool = False, timestamp: bool = False) -> None:
        """
        Saves the model and config to a file

        Args:
            checkpoint: bool - whether to save a checkpoint
            timestamp: bool - whether to timestamp the model (useful when re-training)

        Default save location:
            weights: pcmol/data/models/model_id/model_{checkpoint}.pkg
            config:  pcmol/data/models/model_id/config.yaml
        """

        if checkpoint:
            self.checkpoint += 1
        path = os.path.join(self.config.model_dir, f"model_{self.checkpoint}.pkg")
        if timestamp:
            t = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            path = os.path.join(self.config.model_dir, f"model_{t}.pkg")
        torch.save(self.model.state_dict(), path)
        self.config.save()

    def train(self) -> None:
        """
        Training loop for the model
        All relevant hyperparams are configured in the config file pcmol/pcmol/config.py
        """

        dev = self.config.trainer.dev
        epochs = self.config.trainer.epochs
        batch_size = self.config.trainer.batch_size
        device_ids = self.config.trainer.devices

        parallel = nn.DataParallel(self.model)
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        print(
            f"Model id: {self.model_id}, starting training on {self.config.trainer.dev}:{device_ids}..."
        )
        os.makedirs(self.model_dir, exist_ok=True)

        running_loss = 2.0
        best = float("inf")
        t00 = time.time()
        for epoch in range(epochs):
            n_samples_total = 0
            if self.config.use_wandb:
                wandb.log({"Epoch": epoch})
            print(f"\nEpoch {epoch + 1}/{epochs}\n")

            # Batch loop
            for i, src in enumerate(dataloader):
                proteins, smiles, pchembl = src
                proteins = proteins.to(dev)
                smiles = smiles.squeeze(1).to(dev)
                pchembl = pchembl.to(dev)
                self.optim.zero_grad()

                with torch.cuda.amp.autocast():
                    _, losses = parallel(
                        x=smiles,
                        af_emb=proteins,
                        pchembl_targets=pchembl,
                        calc_pchembl=self.config.model.pcm,
                        train=True,
                    )

                loss = losses
                self.scaler.scale(loss.cuda().mean()).backward()
                self.scaler.step(self.optim)
                self.scaler.update()

                ## Logging
                loss_item = loss.sum().item() / batch_size
                running_loss = 0.98 * running_loss + 0.02 * loss_item
                n_samples_total += batch_size
                percent_complete = n_samples_total / len(self.dataset) * 100

                cl_output = f"Batch {i:4}, running loss: {running_loss:.4f} \
                      | {n_samples_total:8}/{len(self.dataset):8} \
                      | ({percent_complete:.3f}%) | t: {time.time() - t00:.2f}"

                print(cl_output, end="\r")

                if self.config.use_wandb:
                    wandb.log({"running loss": running_loss})
                    wandb.log({"loss": loss_item})
                    wandb.log({"n_samples": n_samples_total})
                    if self.config.model.pcm:
                        pcm_loss = losses["pchembl"].sum().item() / batch_size
                        wandb.log({"loss_pchembl": pcm_loss})
                del loss

                # Saving model / evals
                self.config.trainer.trained_for += batch_size
                if i % self.config.trainer.save_every_n_batches == 0:
                    if running_loss < best:
                        self.save_model()
                        best = running_loss

                # Evaluation
                if i % self.config.trainer.eval_every_n_batches == 0:
                    print("\nEvaluating...")
                    targets = list(self.dataset.proteins.embeddings.keys())
                    df = self.evaluator.test_model()
                    valid = len(df[df.valid == True]) / len(df)
                    unique = len(df[df.unique == True]) / len(df)
                    novel = len(df[df.novel_overall == True]) / len(df)
                    print(f"Valid: {valid}, Unique: {unique}, Novel: {novel}")
                    if self.config.use_wandb:
                        wandb.log(
                            {"Valid %": valid, "Unique %": unique, "Novel %": novel}
                        )
                # Backups
                if i % 10000 == 0:
                    self.save_model(timestamp=True)

            # Save model at the end of each epoch
            self.save_model(checkpoint=True)
        print("Finished training.")

    def targetted_generation(
        self,
        protein_id: str,
        repeat: int = 1,
        batch_size: int = 16,
        verbose: bool = False,
        dev: str = None,
    ) -> pd.DataFrame:
        """
        Generates smiles for a target pid

        Args:
            protein_id: str - target protein ID
            repeat: int - number of times to repeat the generation
            batch_size: int - batch size for generation
            verbose: bool - whether to print the results
            dev: str - device to run on, cuda or cpu

        Returns:
            result_df: pd.DataFrame - dataframe containing the results
        """

        dev = self.config.trainer.dev if dev is None else dev
        net = torch.nn.DataParallel(self.model)

        try:
            ## Check if directory exists
            path = os.path.join(dirs.ALPHAFOLD_DIR, protein_id)
            if not os.path.exists(path):
                # pass
                status = download_protein_data(protein_id)
                if not status:
                    print("Protein embeddings not found...")
                    return None, None
            self.dataset = load_dataset(self.config.dataset, pre_load=True)
        except:
            print("Protein embeddings not found...")

        protein_embedding = self.dataset.proteins.embeddings[protein_id]
        protein_embedding = (
            protein_embedding.unsqueeze(0).repeat(batch_size, 1, 1).to(dev)
        )

        x = torch.LongTensor([[self.voc.tk2ix["GO"]]] * batch_size).to(dev)

        print(f'{"*"*80}\nGenerating smiles for target {protein_id}...')
        smiles_list = []
        with torch.no_grad():
            for i in tqdm(range(repeat)):
                predictions, _ = net(x, af_emb=protein_embedding, train=False)
                smiles = predictions.cpu().numpy()
                for i in range(batch_size):
                    smile = self.voc.decode(smiles[i, :], is_tk=False)
                    smiles_list += [smile]

            scores = check_smiles(smiles_list)

        # Print a summary of the results
        if verbose:
            print(f"Valid smiles: {scores.sum()} / {len(scores)}")
            print(f"Unique smiles: {len(set(smiles_list))} / {len(smiles_list)}")

        ## Save results
        result_df = self.evaluator.evaluate(smiles_list, protein_id, calc_molprops=True)

        ## Save dataframe
        path = os.path.join(dirs.RESULTS_DIR, self.model_id)
        os.makedirs(path, exist_ok=True)
        result_df.to_csv(os.path.join(path, f"results_{protein_id}.csv"))
        print(f"\nSaved results to {path}")
        print(result_df.head(repeat * batch_size))

        return result_df

    def pcm(self, smiles: str, target: str):
        """
        Calculates the pChEMBL value of a list of smiles
        For models that were trained using the pChEMBL loss

        Args:
            smiles: str - smiles string to evaluate
            target: str - target protein uniprot ID

        Returns:
            pchembl: float        - pChEMBL value of the smiles
            gen_smiles: str       - generated smiles
        """

        dev = self.config.trainer.dev
        net = torch.nn.DataParallel(self.model)

        tokenized = smiles.split(" ")[:-1]
        tokenized = self.voc.encode([tokenized]).to(dev)
        bos = torch.LongTensor([[self.voc.tk2ix["GO"]]]).to(dev)
        tokenized = torch.cat([bos, tokenized], dim=1).to(dev)
        target = self.dataset.proteins.embeddings[target].unsqueeze(0).to(dev)

        with torch.no_grad():
            predictions, _ = net(
                tokenized, af_emb=target, train=True, calc_pchembl=True
            )

        pchembl = predictions["pchembl"].cpu().numpy()
        gen_smiles = predictions["tokens"].cpu().numpy()

        return pchembl, gen_smiles

    def evaluate(self, molecules=50, verbose=False, dev="cpu"):
        """
        Evaluates the model by generating N smiles and checking if they are valid
        Extended methods implemented in utils.evaluate.Evaluator class
        """
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=True)
        tensor = torch.LongTensor([[self.voc.tk2ix["GO"]]] * 1).to(dev)
        x = tensor

        ## Load weights from the current model
        self.eval_model.load_state_dict(self.model.state_dict())

        net = nn.DataParallel(self.eval_model)  # , device_ids=self.config.devices
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)

        generated_smiles = []
        lens = []
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                proteins, smiles, pchembl = data
                proteins = proteins.to(dev)
                smiles = smiles.squeeze(1).to(dev)
                pchembl = pchembl.to(dev)

                print(f"Evaluating {i}/{molecules}", end="\r")
                pchembl = pchembl.to(dev)
                predictions, _ = net(x, af_emb=proteins, train=False)

                smiles = predictions["tokens"]
                if i >= molecules:
                    break

                smiles = smiles.squeeze().cpu().numpy()
                try:
                    ind = np.where(smiles == 0)[0][0]
                except:
                    ind = len(smiles)
                cut = smiles[:ind]
                decoded = self.voc.decode(cut, is_tk=False)
                generated_smiles += [decoded]
                lens += [len(decoded)]

        scores = check_smiles(generated_smiles)

        if verbose:
            print("Valid generated smiles:")
        valid, n_valid = [], 0
        for i, smile in enumerate(generated_smiles):
            if scores[i] == 1:
                if smile != "" and smile != " ":
                    valid += [smile]
                    n_valid += 1

        return n_valid / len(scores) * 100, np.mean(lens), valid
