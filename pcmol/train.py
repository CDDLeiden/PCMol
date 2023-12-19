from alphagen.runner import Runner
from alphagen.config import save_config, RunnerConfig
import wandb

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', '-n', type=str, default=None)
    parser.add_argument('--checkpoint', '-c', type=int, default=0)
    parser.add_argument('--use_wandb', '-w', action='store_true')
    parser.add_argument('--project', '-p', type=str, default='alphagen5')

    args = parser.parse_args()

    config = RunnerConfig()
    config.use_wandb = args.use_wandb
    runner = Runner(config, model_id=args.model_id, checkpoint=args.checkpoint)

    if args.use_wandb: 
        wandb.init(project=args.project, entity="cdd-leiden")
        wandb.watch(runner.model)
        wandb.config = save_config(runner)

    runner.train()