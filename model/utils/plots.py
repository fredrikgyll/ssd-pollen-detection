import matplotlib.pyplot as plt
import numpy as np


def make_pr_curves(metrics, classes, with_interpolations=False, save=None):
    fig, axs = plt.subplots(
        ncols=len(classes), sharex=True, sharey=True, figsize=(15, 6)
    )
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

    # dummy axes for axis labels to attach to
    frame = fig.add_subplot(111, frameon=False)
    frame.set_xticks([])
    frame.set_yticks([])
    plt.xlabel("Recall", labelpad=25)
    plt.ylabel("Precision", labelpad=25)

    fig.suptitle('Precision-Recall by class')
    if save:
        plt.savefig(save)
    else:
        plt.show()


def make_map_bars(metrics, classes, save=None):
    y_pos = np.arange(len(classes))
    ap = [metrics[cls]['average_precision'] for cls in classes]

    fig, ax = plt.subplots()
    rects = ax.barh(y_pos, ap, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Average Precision')
    ax.set_title(f'mAP = {np.mean(ap):.2%}')

    bar_labels = [f'{x:.1%}' for x in ap]
    ax.bar_label(rects, labels=bar_labels, padding=3)
    ax.set_xlim(0, 1.1)
    fig.tight_layout()
    if save:
        plt.savefig(save)
    else:
        plt.show()


def make_detection_results_bars(metrics, classes, save=None):
    x_pos = np.arange(len(classes))
    width = 0.35
    gts = [metrics[cls]['ground_truths'] for cls in classes]
    tp = [metrics[cls]['tp'] for cls in classes]
    fp = [metrics[cls]['fp'] for cls in classes]
    dets = [metrics[cls]['total_detections'] for cls in classes]

    fig, ax = plt.subplots()
    rect1 = ax.bar(x_pos - width / 2, gts, width, label='Ground Truth', color='#1f77b4')
    rect2 = ax.bar(x_pos + width / 2, tp, width, label='True Positive', color='#2ca02c')
    rect3 = ax.bar(
        x_pos + width / 2, fp, width, bottom=tp, label='False Positive', color='#d62728'
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(classes)
    ax.set_ylabel('Count')
    ax.set_title('Detection Results')

    ax.bar_label(rect1, label_type='center', color='w')
    ax.bar_label(rect2, label_type='center', color='w')
    ax.bar_label(rect3, label_type='center', color='w')
    ax.bar_label(rect3, padding=2, labels=dets, color='k')
    # ax.set_xlim(0, 1.1)
    fig.tight_layout()
    plt.legend()
    if save:
        plt.savefig(save)
    else:
        plt.show()


def make_run_map(iter, test, train, save=None):

    fig, ax = plt.subplots()
    ax.plot(iter, test, '+:r', label='test')
    ax.plot(iter, train, 'x--g', label='train')
    plt.xlabel('Iteration')
    plt.ylabel('mAP')
    plt.title('Learning mAP')
    plt.legend()
    plt.grid()
    fig.tight_layout()

    if save:
        plt.savefig(save)
    else:
        plt.show()

def make_loss_plot(loss_l, loss_c, save=None):

    fig, ax = plt.subplots()
    ax.scatter(np.arange(len(loss_l)), loss_c+loss_l, s=0.2, marker='.', label='loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Learning Loss')
    plt.grid()
    fig.tight_layout()

    if save:
        plt.savefig(save)
    else:
        plt.show()
