import pickle
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np

def calc_map(evaluation, classes):
    aps = {}
    for n in classes:
        pre = evaluation[n]['precision']
        rec = evaluation[n]['recall']
        pre = np.append(pre, 0.0)
        rec = np.append(rec, 1.0)
        inter = []
        for p in np.linspace(0,1,11):
            inter.append(np.max(pre[rec>=p]))
        aps[n] = inter
    return aps

def make_pr_curve(evaluation, classes, interpolations=None):
    fig, axs = plt.subplots(ncols=len(classes), sharex=True, sharey=True, figsize=(15, 6))
    fig.add_subplot(111, frameon=False)
    inter_x = np.repeat(np.linspace(0, 1, 11), 2)[1:]
    for ax, cls in zip(axs, classes):
        recall = evaluation[cls]['recall']
        precision = evaluation[cls]['precision']
        ax.plot(recall, precision, label=cls)
        if interpolations:
            inter_y = np.repeat(interpolations[cls], 2)[:-1]
            ax.plot(inter_x, inter_y, linestyle='dotted')
        ax.set_title(cls)
        ax.set_xticks(np.linspace(0, 1, 11))
        ax.set_yticks(np.linspace(0, 1, 11))
        ax.grid(which='both')
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    fig.suptitle('Precision-Recall per class')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate mAP from evaluation file')

    parser.add_argument('file', type=Path, help='Path to evaluation pickle')
    parser.add_argument(
        '--plot', '-p', action='store_true', help='Plot the precision-recall curve'
    )

    args = parser.parse_args()

    evaluation = pickle.loads(args.file.read_bytes())
    CLASSES = ['poaceae', 'corylus', 'alnus']
    
    interpolations = calc_map(evaluation, CLASSES)
    aps = [np.mean(interpolations[cls]) for cls in CLASSES]
    for cls, ap in zip(CLASSES, aps):
        print(f'{cls}: {ap:.2%}')
    print(f'mAP: {np.mean(aps):.2%}')

    if args.plot:
        make_pr_curve(evaluation, CLASSES, interpolations=interpolations)
