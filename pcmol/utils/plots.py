import csv
import os
import matplotlib.pyplot as plt
from pcmol.config import BASE_DIR
import pandas as pd
import seaborn as sns


def pdb_to_png(pdb_path):
    """
    Converts a PDB file to a PNG image

    Arguments:
        pdb_path (str) : path to the PDB file
    """
    import __main__
    __main__.pymol_argv = ['pymol','-qc'] # Pymol: quiet and no GUI
    import pymol
    pymol.finish_launching()
    pdb_name = pdb_path.split('.')[0]
    pymol.cmd.load(pdb_path, pdb_name)
    pymol.cmd.disable("all")
    pymol.cmd.enable(pdb_name)
    pymol.cmd.hide('all')
    pymol.cmd.show('cartoon')
    pymol.cmd.set('ray_opaque_background', 0)
    pymol.cmd.color('red', 'ss h')
    pymol.cmd.color('yellow', 'ss s')
    pymol.cmd.png("%s.png"%(pdb_name))
    pymol.cmd.quit()


def plot_training_graph(model_id):
    """
    Plots the training graph for a given model

    Arguments:
        model_num (str) : model id
    """
    if type(model_id) is int:
        model_id = [model_id]

    # Load training history
    losses, valid = [], []
    for model in model_id:
        path = os.path.join(BASE_DIR, 'trained_models', str(model_id), 'training.log')
        with open(path, "r") as f:
            history = csv.reader(f, delimiter="|")
            for i, row in enumerate(history):
                losses += [float(row[1].strip())]
                valid += [float(row[4].strip())]

    plt.plot(losses, label='Training loss')
    plt.plot(valid, label='Valid SMILES Percentage')
    plt.show()


def plot_eval_metrics(model_name, metric, target, xrange=(0,1), every_n=1, start=0):
    """
    Plots the progress of a model
    """
    eval_dir = os.path.join(BASE_DIR, 'alphagen', 'trained_models', model_name, 'evals')
    eval_files = [eval for eval in os.listdir(eval_dir) if eval.endswith('.csv')]
    # List by date of creation
    eval_files = sorted(eval_files, key=lambda x: os.path.getctime(os.path.join(eval_dir, x)))

    # Load evaluation history 
    evals = pd.DataFrame()
    for i, eval_file in enumerate(eval_files[start::every_n]):
        path = os.path.join(eval_dir, eval_file)
        eval_df = pd.read_csv(path)
        try:
            eval_df = eval_df[eval_df.target == target]
        except:
            pass
        # eval_df = eval_df[metric].to_frame()
        eval_df['eval_index'] = i
        evals = pd.concat([evals, eval_df])
    
    # Replace 0s with NaNs
    evals = evals.replace(0, float('nan'))

    print('Plotting...')
    # Decrease plot size
    ## SNS stacked ridge plot
    
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    palette = sns.cubehelix_palette(10, start=.5, rot=-.75)
    sns.set_palette(palette)
    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(i+1, rot=-.25, light=.7)
    g = sns.FacetGrid(evals, row="eval_index", hue="eval_index", aspect=15, height=.25, palette=pal)
    # Draw the densities in a few steps
    g.map(sns.kdeplot, metric, clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
    g.map(sns.kdeplot, metric, clip_on=False, color="w", lw=1, bw=.2)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)
    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(1.05, 0, label, fontsize=7, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)
    g.map(label, metric)
    # X lim
    # Plot vertical line at 0.5
    plt.axvline(x=0.5, color='black', linestyle='--')
    plt.xlim(*xrange)

    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-.98)
    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.set(ylabel="")
    g.despine(bottom=True, left=True)
    plt.rcParams['figure.figsize'] = [4, 6]
    plt.show()

    return evals


