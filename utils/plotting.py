import matplotlib as mpl
import matplotlib.pyplot as plt
from optim.utils import read_all_runs
import numpy as np
import os

plt.style.use('ggplot')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['legend.fontsize'] = 'x-large'
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['axes.labelsize'] = 'x-large'

PLOT_PATH = '/kaggle/working/exps_setup/plots/'


def plot(exps, kind, suffix=None, log_scale=True, legend=None, file=None,
         x_label='epochs', y_label=None, title=None):
    fig, ax = plt.subplots()

    for exp, lab in zip(exps, legend):
        runs = read_all_runs(exp, suffix=suffix)
        plot_mean_std(ax, runs, kind, lab)

    if log_scale:
        ax.set_yscale('log')
    #if legend is not None:
    #    ax.legend(legend)
    plt.legend()
    if title is not None:
        plt.title(title)

    ax.set_xlabel(x_label)
    if y_label is None:
        ax.set_ylabel(kind)
    else:
        ax.set_ylabel(y_label)

    fig.tight_layout()
    if file is not None:
        if not os.path.isdir(PLOT_PATH):
            os.mkdir(PLOT_PATH)
        plt.savefig(PLOT_PATH + file + '.pdf')
    plt.show()


def plot_mean_std(ax, runs, kind, lab):
    quant = np.array([run[kind] for run in runs])
    mean = np.mean(quant, axis=0)
    std = np.std(quant, axis=0)
    mean = np.array([mean]) if not isinstance(mean, np.ndarray) else mean
    x = np.arange(1, len(mean) + 1)
    ax.plot(x, mean, label=lab)
    #ax.fill_between(x, mean + std, mean - std, alpha=0.4)
