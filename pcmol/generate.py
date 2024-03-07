import os
from pcmol.models.runner import Runner
from pcmol.config import dirs
import pandas as pd

## Suppress RDKit warnings
from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="XL")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on, cuda or cpu",
    )
    parser.add_argument("--device_id", type=int, default=0, help="CUDA device ID")
    parser.add_argument(
        "--start", type=int, default=0, help="Start index for protein IDs"
    )
    parser.add_argument("--end", type=int, default=-1, help="End index for protein IDs")
    parser.add_argument(
        "--checkpoint", type=int, default=7, help="Checkpoint number of the model file"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="TSV file with target IDs in the first column",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default=None,
        help="TSV file with target IDs in the first column",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for generation"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=10,
        help="Number of times to repeat generation. \
                        Total number of samples is repeats*batch_size.",
    )
    args = parser.parse_args()

    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    trainer = Runner(
        model_id=args.model,
        checkpoint=args.checkpoint,
        device=args.device,
        inference=True,
    )

    if args.targets is not None:
        pids = [args.targets]
    else:
        if args.file is None:
            dataset_path = os.path.join(
                dirs.DATA_DIR, "final_augmented", "unaugmented.tsv"
            )
            papyrus = pd.read_csv(dataset_path, sep="\t")
            pids = papyrus["target_id"].unique()
            pids = pids[args.start : args.end]
        else:
            if args.targets.endswith(".tsv") or args.targets.endswith(".csv"):
                papyrus = pd.read_csv(args.targets, sep="\t")
                pids = papyrus["target_id"].unique()
            else:
                with open(args.targets, "r") as f:
                    pids = f.read().splitlines()

    for i, pid in enumerate(pids):
        print("\n", i, pid)

        smiles_df = trainer.targetted_generation(
            protein_id=pid,
            batch_size=1,
            repeat=args.repeat,
            verbose=True,
            dev=args.device,
        )
