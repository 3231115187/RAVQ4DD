import torch
import torch.nn.functional as F
import numpy as np
from data_utils import eval_f1_micro_macro


@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args, node_type, device):
    model.eval()

    id_out, aux_out, gnn_out, _ = model(
        dataset.graph['node_feat'],
        dataset.graph['edge_index'],
        node_type,
        vq_noise_scale=0.0
    )

    out = id_out

    train_score = eval_func(dataset.label[split_idx['train']], out[split_idx['train']])
    valid_score = eval_func(dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_score = eval_func(dataset.label[split_idx['test']], out[split_idx['test']])

    val_micro, val_macro = eval_f1_micro_macro(
        dataset.label[split_idx['valid']], out[split_idx['valid']]
    )
    test_micro, test_macro = eval_f1_micro_macro(
        dataset.label[split_idx['test']], out[split_idx['test']]
    )

    if args.dataset in ('questions'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[split_idx['valid']].to(torch.float))
    else:
        if isinstance(criterion, torch.nn.NLLLoss):
            out_log = F.log_softmax(out, dim=1)
            valid_loss = criterion(out_log[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])
        else:
            valid_loss = criterion(out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_score, valid_score, test_score, valid_loss, val_micro, val_macro, test_micro, test_macro