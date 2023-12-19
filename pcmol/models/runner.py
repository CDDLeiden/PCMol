import torch
import wandb
import os
import time
import datetime
import numpy as np
from tqdm import tqdm
from torch import nn
from pcmol.models import AF2SmilesTransformer, count_parameters
from pcmol.utils.smiles import check_smiles
from torch.utils.data import DataLoader
from pcmol.utils.dataset import load_dataset, load_voc
from pcmol.config import RunnerConfig, load_config, save_config, MODEL_DIR
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

    def __init__(self, config: RunnerConfig=None, model_id: str=None, 
                 checkpoint: int=0, load: bool=False, device: str='cuda') -> None:

        if model_id is not None:
            # base_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(MODEL_DIR, str(model_id))
            print(model_dir)
            if not os.path.exists(model_dir):
                self.id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            else:
                load = True
                config_path = os.path.join(model_dir, 'config.yaml')
                config = self.config = load_config(config_path)
                config.model_dir = model_dir
        self.config = config
        self.config.trainer.dev = torch.device(device)

        # Model ID
        model_dir = MODEL_DIR
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model_id = str(model_id) + str(self.timestamp)
        self.parent_model = model_id
        self.model_dir = os.path.join(model_dir, self.model_id)
        self.config.model_dir = self.model_dir
        self.config.model_id = self.model_id

        self.voc = load_voc(config.dataset)
        self.dataset = load_dataset(config.dataset, pre_load=True)
        self.model = AF2SmilesTransformer(self.voc, **config.model.__dict__, dev=device)
        self.evaluator = Evaluator(self,
                                   config.evaluator,
                                   use_wandb=config.use_wandb)
        self.checkpoint = checkpoint

        if load:
            weights_file = os.path.join(MODEL_DIR, model_id,
                                        f'model_{self.checkpoint}.pkg')
            self.model.load_state_dict(
                torch.load(weights_file, map_location=self.config.trainer.dev))

        self.optim = torch.optim.AdamW(self.model.parameters(),
                                       lr=config.trainer.lr,
                                       betas=(0.95, 0.98))
        
        self.scaler = torch.cuda.amp.GradScaler()
        param_count = count_parameters(self.model)
        self.config.trainer.parameter_count = param_count

        print(f'Loaded model {model_id}, parameter count: {param_count} \
              \nModel directory: {config.model_dir}, checkpoint: {checkpoint}')

    def save_model(self, checkpoint: bool = False, timestamp: bool = False):
        """ 
        Saves the model and config to a file 
        """
        if checkpoint:
            self.checkpoint += 1
        path = os.path.join(self.config.model_dir,
                            f'model_{self.checkpoint}.pkg')
        if timestamp:
            t = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            path = os.path.join(self.config.model_dir, f'model_{t}.pkg')
        torch.save(self.model.state_dict(), path)
        save_config(self)

    def train(self, epochs: int = -1):
        """
        Training loop
        """
        epochs = self.config.trainer.epochs if epochs == -1 else epochs
        batch_size = self.config.trainer.batch_size
        dev = self.config.trainer.dev
        device_ids = self.config.trainer.devices

        os.makedirs(self.model_dir, exist_ok=True)
        print(device_ids)

        self.parallel = nn.DataParallel(self.model)
        dataloader = DataLoader(self.dataset,
                                batch_size=batch_size,
                                shuffle=True)

        print(f'Model id: {self.model_id}, starting training...')
        running_loss = 2.0
        best = float('inf')
        t00 = time.time()
        for epoch in range(epochs):
            n_samples_total = 0
            if self.config.use_wandb:
                wandb.log({'Epoch': epoch})
            print(f'\nEpoch {epoch + 1}/{epochs}\n')

            for i, src in enumerate(dataloader):
                proteins, smiles, pchembl = src
                proteins = proteins.to(dev)
                smiles = smiles.squeeze(1).to(dev)
                pchembl = pchembl.to(dev)
                self.optim.zero_grad()

                with torch.cuda.amp.autocast():
                    _, losses = self.parallel(
                        x=smiles,
                        af_emb=proteins,
                        pchembl_targets=pchembl,
                        calc_pchembl=self.config.model.pcm,
                        train=True)

                loss = losses

                self.scaler.scale(loss.cuda().mean()).backward()
                self.scaler.step(self.optim)
                self.scaler.update()

                ## Logging
                loss_item = loss.sum().item() / batch_size
                running_loss = .98 * running_loss + .02 * loss_item
                n_samples_total += batch_size
                percent_complete = n_samples_total / len(self.dataset) * 100

                cl_output = f'Batch {i:4}, running loss: {running_loss:.4f} \
                      | {n_samples_total:8}/{len(self.dataset):8} \
                      | ({percent_complete:.3f}%) | t: {time.time() - t00:.2f}'

                print(cl_output, end='\r')

                if self.config.use_wandb:
                    wandb.log({'running loss': running_loss})
                    wandb.log({'loss': loss_item})
                    wandb.log({'n_samples': n_samples_total})
                    if self.config.model.pcm:
                        pcm_loss = losses['pchembl'].sum().item() / batch_size
                        wandb.log({'loss_pchembl': pcm_loss})
                del loss

                # Saving model / evals
                self.config.trainer.trained_for += batch_size
                if i % self.config.trainer.save_every_n_batches == 0:
                    if running_loss < best:
                        self.save_model()
                        best = running_loss

                # Evaluation
                if i % self.config.trainer.eval_every_n_batches == 0:
                    print('\nEvaluating...')
                    targets = list(self.dataset.proteins.embeddings.keys())
                    df = self.evaluator.test_model()
                    valid = len(df[df.valid == True]) / len(df)
                    unique = len(df[df.unique == True]) / len(df)
                    novel = len(df[df.novel_overall == True]) / len(df)
                    print(f'Valid: {valid}, Unique: {unique}, Novel: {novel}')
                    if self.config.use_wandb:
                        wandb.log({
                            'Valid %': valid,
                            'Unique %': unique,
                            'Novel %': novel
                        })
                # Backups
                if i % 10000 == 0:
                    self.save_model(timestamp=True)
            self.save_model(checkpoint=True)

    def targetted_generation(self, protein_id, repeat=1, batch_size=16, verbose=False):
        """
        Generates smiles for a target pid
        """
        dev = self.config.trainer.dev
        net = torch.nn.DataParallel(self.model)

        try:
            self.dataset.proteins.embeddings[protein_id]
        except:
            status = download_protein_data(protein_id)
            if status:
                self.dataset = load_dataset(self.config.dataset, pre_load=True)
            else:
                print('Protein embeddings not found...')
        

        protein_embedding = self.dataset.proteins.embeddings[protein_id]
        protein_embedding = protein_embedding.unsqueeze(0).repeat(
            batch_size, 1, 1).to(dev)

        x = torch.LongTensor([[self.voc.tk2ix['GO']]] * batch_size).to(dev)

        smiles_list = []
        with torch.no_grad():
            for i in tqdm(range(repeat)):
                # print(f'Generating SMILES {(i+1)*batch_size}/{repeat*batch_size}', end='\r')

                predictions, _ = net(x, af_emb=protein_embedding, train=False)
                smiles = predictions.cpu().numpy()
                for i in range(batch_size):
                    smile = self.voc.decode(smiles[i, :], is_tk=False)
                    smiles_list += [smile]

            scores = check_smiles(smiles_list)

        # Print a summary of the results
        if verbose:
            print(f'Valid smiles: {scores.sum()} / {len(scores)}')
            print(f'Unique smiles: {len(set(smiles_list))} / {len(smiles_list)}')

            print('Valid smiles:\n')
            for smile in smiles_list:
                if check_smiles([smile])[0] == 1:
                    print(smile)

        return smiles_list, scores.sum() / batch_size * repeat

    def pcm(self, smiles, target):
        """
        Calculates the pChEMBL value of a list of smiles
        For models that were trained using the pChEMBL loss
        """
        dev = self.config.trainer.dev
        net = torch.nn.DataParallel(self.model)

        tokenized = smiles.split(' ')[:-1]
        tokenized = self.voc.encode([tokenized]).to(dev)
        bos = torch.LongTensor([[self.voc.tk2ix['GO']]]).to(dev)
        tokenized = torch.cat([bos, tokenized], dim=1).to(dev)
        target = self.dataset.proteins.embeddings[target].unsqueeze(0).to(dev)

        with torch.no_grad():
            predictions, _ = net(tokenized,
                                 af_emb=target,
                                 train=True,
                                 calc_pchembl=True)

        pchembl = predictions['pchembl'].cpu().numpy()
        gen_smiles = predictions['tokens'].cpu().numpy()

        return pchembl, gen_smiles

    def evaluate(self, molecules=50, verbose=False, dev='cpu'):
        """
        Evaluates the model by generating N smiles and checking if they are valid
        Extended methods implemented in utils.evaluate.Evaluator class
        """
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=True)
        tensor = torch.LongTensor([[self.voc.tk2ix['GO']]] * 1).to(dev)
        x = tensor

        ## Load weights from the current model
        self.eval_model.load_state_dict(self.model.state_dict())

        net = nn.DataParallel(
            self.eval_model)  #, device_ids=self.config.devices
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)

        generated_smiles = []
        lens = []
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                proteins, smiles, pchembl = data
                proteins = proteins.to(dev)
                smiles = smiles.squeeze(1).to(dev)
                pchembl = pchembl.to(dev)

                print(f'Evaluating {i}/{molecules}', end='\r')
                pchembl = pchembl.to(dev)
                predictions, _ = net(x, af_emb=proteins, train=False)

                smiles = predictions['tokens']
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
                if smile != '' and smile != ' ':
                    valid += [smile]
                    n_valid += 1

        return n_valid / len(scores) * 100, np.mean(lens), valid
