import matplotlib.pyplot as plt
import numpy as np


def make_pr_curves(metrics, classes, with_interpolations=False):
    fig, axs = plt.subplots(
        ncols=len(classes), sharex=True, sharey=True, figsize=(15, 6)
    )
    fig.add_subplot(111, frameon=False)
    inter_x = np.repeat(np.linspace(0, 1, 11), 2)[1:]
    for ax, cls in zip(axs, classes):
        recall = metrics[cls]['recall']
        precision = metrics[cls]['precision']
        ax.plot(recall, precision, label=cls)
        if with_interpolations:
            inter_y = np.repeat(metrics[cls]['interpolation'], 2)[:-1]
            ax.plot(inter_x, inter_y, linestyle='dotted')
        ax.set_title(cls)
        ax.set_xticks(np.linspace(0, 1, 11))
        ax.set_yticks(np.linspace(0, 1, 11))
        ax.grid(which='both')
    plt.tick_params(
        labelcolor='none',
        which='both',
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    fig.suptitle('Precision-Recall per class')
    plt.show()


def make_map_bars(metrics, classes):
    y_pos = np.arange(len(classes))
    ap = [metrics[cls]['average_precision'] for cls in classes]

    plt.barh(y_pos, ap, align='center')
    plt.yticks(y_pos, classes)
    plt.xlabel('Average Precision')
    plt.title(f'mAP = {np.mean(ap):.2%}')

    plt.show()
