# Model, Dataloader, Ground truth
import torch

from utils.geometry import decode, jaccard


def evaluate(model, data_loader, args):
    model.eval()
    confidences = []
    predictions = []
    nr_gts = 0
    for i, batch in enumerate(data_loader):
        print(f'Batch {i:2d}')
        images, targets, labels = batch.data()
        images = images.cuda()
        targets = [t.cuda() for t in targets]
        labels = [l.cuda() for l in labels]

        with torch.no_grad():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            ploc, pconf = model(images)
            pboxes, confs, _ = decode(model.priors, ploc, pconf)  # TODO: per label
            for box, truth, conf in zip(pboxes, targets, confs):
                nr_gts += truth.size(0)
                sorted_conf, order_idx = conf.sort(descending=True)
                iou = jaccard(box[order_idx], truth)
                preds = torch.zeros(box.size(0))
                for ground_column in iou.split(1, 1):
                    tp = torch.nonzero(ground_column.squeeze() > 0.5).squeeze(-1)
                    if tp.nelement() > 0:
                        preds[tp[0]] = 1
                predictions.append(preds)
                confidences.append(sorted_conf)
    predictions = torch.cat(predictions, dim=0)
    confidences = torch.cat(confidences, dim=0)
    _, order = confidences.sort(descending=True)
    predictions = torch.cumsum(predictions[order], dim=0)
    precision = predictions / torch.arange(1, predictions.nelement() + 1)
    recall = predictions / nr_gts
    return precision, recall
