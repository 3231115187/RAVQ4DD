import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import eval_acc, eval_rocauc, load_fixed_splits, class_rand_splits, eval_f1_micro_macro
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, dropout_edge

from logger import Logger, save_model, save_result
from dataset import load_dataset
from model import RAVQ4DD
from parse import parser_add_main_args


import matplotlib.pyplot as plt
import seaborn as sns
import os


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1).long(), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


@torch.no_grad()
def evaluate_relation_aware(model, dataset, split_idx, eval_func, criterion, args, node_type, device):
    model.eval()

    id_out, aux_out, gnn_out, _, _ = model(
        dataset.graph['node_feat'],
        dataset.graph['edge_index'],
        dataset.graph['edge_type'],
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



class CodebookAnalysis:
    def __init__(self, model, dataset, device, args):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.args = args
        self.num_classes = dataset.label.max().item() + 1

        # 疾病名称映射
        self.disease_map = {
            0: 'Heart Failure',
            1: 'Myocardial',
            2: 'Cirrhosis',
            3: 'Acute Pancreatitis',
            4: 'Pneumonia',
            5: 'Hypertension'
        }

    @torch.no_grad()
    def run_analysis(self):
        print(f"\n{bcolors.HEADER}=== Starting Deep Case Study & Clean Heatmap Generation ==={bcolors.ENDC}")
        self.model.eval()

        node_type = self.dataset.graph.get('node_type', torch.zeros(self.dataset.graph['num_nodes'])).to(self.device)
        edge_type = self.dataset.graph['edge_type'].to(self.device)

        id_out, aux_out, gnn_out, _, all_indices = self.model(
            self.dataset.graph['node_feat'],
            self.dataset.graph['edge_index'],
            edge_type,
            node_type,
            vq_noise_scale=0.0
        )

        pred_prob = F.softmax(id_out, dim=1)
        pred_label = pred_prob.argmax(dim=1)
        true_label = self.dataset.label.squeeze()
        test_idx = self.dataset.test_idx

        if all_indices is not None:
            heatmap_matrix = torch.zeros((self.num_classes, self.args.num_codes), device=self.device)
            class_counts = torch.zeros(self.num_classes, device=self.device)

            print(f"Collecting statistics from {len(test_idx)} test samples...")
            for idx in test_idx:
                label = true_label[idx].item()
                class_counts[label] += 1

                codes = all_indices[idx].flatten()
                unique_codes = torch.unique(codes)
                for code in unique_codes:
                    heatmap_matrix[label, code] += 1

            norm_matrix = heatmap_matrix / (class_counts.unsqueeze(1) + 1e-6)

            filtered_matrix, top_codes = self.plot_clean_heatmap(norm_matrix)

            self.perform_case_study(test_idx, pred_label, true_label, pred_prob, all_indices, norm_matrix)
        else:
            print("Warning: No VQ indices returned.")

    def plot_clean_heatmap(self, prob_matrix):
        print("Generating Optimized 'Elite Codes' Heatmap (Extra Large Fonts)...")

        #挑选码字
        top_k = 6
        elite_codes = set()
        class_to_codes = {}

        for c in range(self.num_classes):
            _, indices = torch.topk(prob_matrix[c], k=top_k)
            indices = indices.cpu().numpy().tolist()
            #过滤掉低频噪音
            valid_indices = [idx for idx in indices if prob_matrix[c, idx] > 0.05]
            class_to_codes[c] = valid_indices
            elite_codes.update(valid_indices)

        #排序
        sorted_indices = []
        for c in range(self.num_classes):
            for code in class_to_codes[c]:
                if code not in sorted_indices:
                    sorted_indices.append(code)

        #提取子矩阵
        final_matrix = prob_matrix[:, sorted_indices].cpu().numpy()

        plt.figure(figsize=(20, 12))

        #绘制热力图
        ax = sns.heatmap(final_matrix, cmap="Blues", annot=False,
                         vmin=0.0, vmax=1.0,
                         cbar=True,
                         cbar_kws={'label': 'Activation Frequency'})


        cbar = ax.collections[0].colorbar

        cbar.ax.tick_params(labelsize=24)

        cbar.set_label('Activation Frequency', size=28)

        yticks_labels = [self.disease_map.get(i, f'Class {i}') for i in range(self.num_classes)]
        plt.yticks(np.arange(self.num_classes) + 0.5,
                   yticks_labels,
                   rotation=0,
                   fontsize=24)
        plt.xticks(np.arange(len(sorted_indices)) + 0.5,
                   sorted_indices,
                   rotation=0,
                   fontsize=20)

        save_path = f'results/{self.args.dataset}/heatmap_elite.png'
        if not os.path.exists(os.path.dirname(save_path)): os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{bcolors.OKGREEN}Clean Heatmap saved to: {os.path.abspath(save_path)}{bcolors.ENDC}")

        return final_matrix, sorted_indices

    def perform_case_study(self, test_idx, pred_label, true_label, pred_prob, all_indices, norm_matrix):
        print(f"\n{bcolors.HEADER}--- Deep Case Study: Semantic Path Tracking & Verification ---{bcolors.ENDC}")

        candidates = []
        for idx in test_idx:
            if pred_label[idx] == true_label[idx]:
                confidence = pred_prob[idx, pred_label[idx]].item()
                candidates.append((idx.item(), confidence))

        candidates.sort(key=lambda x: x[1], reverse=True)
        top_cases = candidates[:3]

        for i, (node_idx, conf) in enumerate(top_cases):
            disease_idx = true_label[node_idx].item()
            disease_name = self.disease_map.get(disease_idx, f"Class {disease_idx}")

            code_seq = all_indices[node_idx].cpu().numpy()

            print(f"\n{bcolors.BOLD}[Case #{i + 1}] Patient Node {node_idx}{bcolors.ENDC}")
            print(f"  > True Disease:   {disease_name} (ID: {disease_idx})")
            print(f"  > Model Predict:  {disease_name} (Confidence: {conf:.2%})")

            print(f"  > {bcolors.OKBLUE}Diagnostic Path with Activation Evidence:{bcolors.ENDC}")

            total_codes = len(code_seq)
            num_layers = self.args.local_layers if hasattr(self.args, 'local_layers') else 3
            steps_per_layer = total_codes // num_layers

            history_codes = set()

            for l in range(num_layers):
                start = l * steps_per_layer
                end = (l + 1) * steps_per_layer
                layer_codes = code_seq[start:end]

                evidence = []
                for code in layer_codes:
                    c_val = int(code.item())

                    # 获取频率
                    prob = norm_matrix[disease_idx, c_val].item()
                    prob_pct = f"{prob:.1%}"

                    strength_tag = ""
                    if prob > 0.5:
                        strength_tag = "Strong"
                    elif prob > 0.3:
                        strength_tag = "Moderate"

                    consistency = ""
                    if c_val in history_codes:
                        consistency = "[Consistent]"
                    history_codes.add(c_val)

                    if strength_tag or consistency:
                        info_str = f"Activation Freq: {prob_pct}"
                        if strength_tag: info_str += f" - {strength_tag} Evidence!"
                        evidence.append(f"Code {c_val} ({info_str}){consistency}")
                    else:
                        evidence.append(f"Code {c_val} (Freq: {prob_pct})")

                ev_str = " | ".join(evidence) if evidence else "General Features"
                print(f"    - Layer {l + 1}: {ev_str}")

            print(f"  > Verification:")
            print(f"    Please check the Heatmap x-axis. The codes listed above with high 'Activation Freq'")
            print(f"    should correspond to the dark blocks in the '{disease_name}' row.")


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Pipeline')

    parser_add_main_args(parser)

    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--early_stopping', action='store_true', default=True)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    args = parser.parse_args()
    print(f"Running Experiment: {args.method} on {args.dataset}")

    fix_seed(args.seed)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() and not args.cpu else torch.device(
        "cpu")

    #加载数据
    dataset = load_dataset(args.data_dir, args.dataset)
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    dataset.label = dataset.label.to(device)

    #节点类型
    if 'node_type' in dataset.graph:
        node_type = dataset.graph['node_type'].to(device)
        num_node_types = int(node_type.max()) + 1
    else:
        n = dataset.graph['num_nodes']
        node_type = torch.zeros(n, dtype=torch.long).to(device)
        num_node_types = 1
    args.real_num_node_types = num_node_types

    #关系类型
    if 'edge_type' not in dataset.graph:
        print(f"{bcolors.WARNING}Warning: 'edge_type' not found. Using default.{bcolors.ENDC}")
        dataset.graph['edge_type'] = torch.zeros(dataset.graph['edge_index'].shape[1], dtype=torch.long)

    dataset.graph['edge_type'] = dataset.graph['edge_type'].to(device)
    num_relations = int(dataset.graph['edge_type'].max()) + 1
    args.real_num_relations = num_relations
    print(f"Detected {num_relations} relation types.")

    if args.rand_split:
        split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop) for _ in
                         range(args.runs)]
    elif args.rand_split_class:
        split_idx_lst = [class_rand_splits(dataset.label, args.label_num_per_class, args.valid_num, args.test_num)]
    else:
        split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

    n = dataset.graph['num_nodes']
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]

    dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device)
    dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)

    print(f"Dataset Info: Nodes={n}, Classes={c}, Features={d}, Relations={num_relations}")

    # 初始化
    from parse import parse_method

    model = parse_method(args, n, c, d, device)

    # 如果加载预训练模型
    if args.load_pretrained:
        print(f"Loading pretrained model from {args.load_pretrained}...")
        checkpoint = torch.load(args.load_pretrained)
        model.load_state_dict(checkpoint['model_state_dict'])
        dataset.train_idx = split_idx_lst[0]['train']
        dataset.valid_idx = split_idx_lst[0]['valid']
        dataset.test_idx = split_idx_lst[0]['test']
        analyzer = CodebookAnalysis(model, dataset, device, args)
        analyzer.run_analysis()
        exit()

    if args.dataset in ('questions'):
        criterion = nn.BCEWithLogitsLoss()
    else:
        if args.label_smoothing > 0:
            criterion = LabelSmoothingLoss(classes=c, smoothing=args.label_smoothing)
        else:
            criterion = nn.NLLLoss()

    eval_func = eval_rocauc if args.metric == 'rocauc' else eval_acc
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        split_idx = split_idx_lst[run % len(split_idx_lst)]
        train_idx = split_idx['train'].to(device)

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=30)

        best_val = float('-inf')
        best_test_acc = 0
        patience_counter = 0
        best_epoch = 0

        print(f"\n=== Run {run} Start ===")

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()

            if args.drop_edge_rate > 0:
                edge_index_train, edge_mask = dropout_edge(
                    dataset.graph['edge_index'],
                    p=args.drop_edge_rate,
                    force_undirected=False,
                    training=True
                )
                edge_type_train = dataset.graph['edge_type'][edge_mask]
            else:
                edge_index_train = dataset.graph['edge_index']
                edge_type_train = dataset.graph['edge_type']

            id_out, aux_out, gnn_out, commit_loss, _ = model(
                dataset.graph['node_feat'],
                edge_index_train,
                edge_type_train,
                node_type,
                vq_noise_scale=args.vq_noise_scale
            )

            target = dataset.label.squeeze(1)[train_idx]

            if isinstance(criterion, LabelSmoothingLoss):
                loss_id = criterion(id_out[train_idx], target)
                loss_aux = criterion(aux_out[train_idx], target)
                loss_gnn = criterion(gnn_out[train_idx], target)
            else:
                loss_id = criterion(F.log_softmax(id_out[train_idx], dim=1), target)
                loss_aux = criterion(F.log_softmax(aux_out[train_idx], dim=1), target)
                loss_gnn = criterion(F.log_softmax(gnn_out[train_idx], dim=1), target)

            current_gnn_weight = args.gnn_loss_weight
            if epoch > args.epochs * 0.5:
                current_gnn_weight = current_gnn_weight * 0.1

            vq_beta = 0.5

            total_loss = loss_id + \
                         (args.aux_loss_weight * loss_aux) + \
                         (current_gnn_weight * loss_gnn) + \
                         (vq_beta * commit_loss)

            total_loss.backward()
            optimizer.step()

            result = evaluate_relation_aware(model, dataset, split_idx, eval_func, criterion, args, node_type, device)

            scheduler.step(result[1])
            logger.add_result(run, result)

            if result[1] > best_val:
                best_val = result[1]
                best_test_acc = result[2]
                best_epoch = epoch
                patience_counter = 0
                if args.save_model:
                    save_model(args, model, optimizer, run)
            else:
                patience_counter += 1

            if epoch % args.display_step == 0:
                print(f'Epoch: {epoch:02d}, Loss: {total_loss:.4f} | '
                      f'Val Acc: {100 * result[1]:.2f}%, Val Mac: {100 * result[5]:.2f}% | '
                      f'Test Mic: {100 * result[6]:.2f}%, Test Mac: {100 * result[7]:.2f}%')

        print(
            f'Run {run} End. Best Epoch: {best_epoch}, Best Val: {100 * best_val:.2f}%, Test Acc: {100 * best_test_acc:.2f}%')
        logger.print_statistics(run)

    print("\n=== Final Results ===")
    results = logger.print_statistics()
    if args.save_result:
        save_result(args, results)


    if args.analysis:
        print("Reloading best model for analysis...")
        best_model_path = f'models/{args.dataset}/{args.method}_{args.runs - 1}.pt'
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])

        dataset.train_idx = split_idx_lst[-1]['train']
        dataset.valid_idx = split_idx_lst[-1]['valid']
        dataset.test_idx = split_idx_lst[-1]['test']

        analyzer = CodebookAnalysis(model, dataset, device, args)
        analyzer.run_analysis()