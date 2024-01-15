from pcmol.models.runner import Runner
import os
from pcmol.config import dirs

## Suppress RDKit warnings
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


class Generator(Runner):
    pass



# def generate(protein_id, model='XL', checkpoint=7, device='cuda', repeat=10):
#     trainer = Runner(model, checkpoint=checkpoint, device=device)
#     smiles, _ = trainer.targetted_generation(protein_id=protein_id, batch_size=1, repeat=repeat, verbose=True)
#     return smiles

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='XL')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--checkpoint', type=int, default=7)
    parser.add_argument('--file', type=str, default=None)
    parser.add_argument('--targets', type=str, default=None)
    parser.add_argument('--repeat', type=int, default=10)
    args = parser.parse_args()
    import os

    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES']=str(args.device_id)
    else:
        os.environ['CUDA_VISIBLE_DEVICES']=''

    # main(model_num=args.model, start=args.start, end=args.end)

    import pandas as pd

    trainer = Runner(model_id=args.model, checkpoint=args.checkpoint, device=args.device)

    if args.targets is not None:
        pids = [args.targets]
    else:
        if args.file is None:
            path = os.path.join(dirs.DATA_DIR, 'final_augmented', 'unaugmented.tsv')
            papyrus = pd.read_csv(path, sep='\t')
            pids = papyrus['target_id'].unique()
            pids = pids[args.start:args.end]
        else:
            if args.targets.endswith('.tsv') or args.targets.endswith('.csv'):
                papyrus = pd.read_csv(args.targets, sep='\t')
                pids = papyrus['target_id'].unique()
            else:
                with open(args.targets, 'r') as f:
                    pids = f.read().splitlines()
    
    for i, pid in enumerate(pids):
        print('\n', i, pid)

        smiles_df, _ = trainer.targetted_generation(protein_id=pid, batch_size=1, repeat=args.repeat, verbose=True, dev=args.device)
