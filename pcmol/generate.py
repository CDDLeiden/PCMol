from pcmol.runner import Runner
import os

## Suppress RDKit warnings
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def check_existing(pid):
    data_dir = '/home/andrius/datasets/output_l'
    subdir = os.path.join(data_dir, pid)
    smiles_out = os.path.join(subdir, f'{pid}.txt')
    return os.path.exists(smiles_out)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='XL')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--checkpoint', type=int, default=0)
    parser.add_argument('--file', type=str, default=None)
    parser.add_argument('--targets', type=str, default=None)
    parser.add_argument('--repeat', type=int, default=4)
    args = parser.parse_args()
    import os
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.device)

    # main(model_num=args.model, start=args.start, end=args.end)

    import pandas as pd

    trainer = Runner(model_id=args.model, checkpoint=args.checkpoint)

    if args.file is None:
        papyrus = pd.read_csv('/home/andrius/datasets/final_augmented/unaugmented.tsv', sep='\t')
        pids = papyrus['target_id'].unique()
        pids = pids[args.start:args.end]
    else:
        if args.targets.endswith('.tsv') or args.targets.endswith('.csv'):
            papyrus = pd.read_csv(args.targets, sep='\t')
            pids = papyrus['target_id'].unique()
        else:
            with open(args.targets, 'r') as f:
                pids = f.read().splitlines()
    
    if args.targets is not None:
        pids = [args.targets]

    data_dir = f'/home/andrius/alphagen/data/outputs/output_{args.model}'
    os.makedirs(data_dir, exist_ok=True)
    for i, pid in enumerate(pids):
        print('\n', i, pid)
        if check_existing(pid):
            print('Output already exists')
            continue
        smiles, _ = trainer.targetted_generation(protein_id=pid, batch_size=1, repeat=args.repeat)
        # papyrus.loc[papyrus['target_id'] == pid, 'smiles'] = smiles

        subdir = os.path.join(data_dir, pid)
        os.makedirs(subdir, exist_ok=True)
        smiles_out = os.path.join(subdir, f'{pid}.txt')
        
        ## Write smiles as a text file
        with open(smiles_out, 'a') as f:
            f.write("\n".join(smiles))