import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from vq import DynamicResidualVectorQuant


class TypeSpecificProjector(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_node_types):
        super().__init__()
        self.num_node_types = num_node_types
        self.projectors = nn.ModuleList([
            nn.Linear(in_channels, hidden_channels, bias=True) for _ in range(num_node_types)
        ])

    def reset_parameters(self):
        for proj in self.projectors:
            proj.reset_parameters()

    def forward(self, x, node_type):
        out = torch.zeros(x.size(0), self.projectors[0].out_features, device=x.device)
        for i in range(self.num_node_types):
            mask = (node_type == i)
            if mask.any():
                out[mask] = self.projectors[i](x[mask])
        return out


class RelationAwareLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, heads=1, dropout=0.5, **kwargs):
        super().__init__(aggr='add', node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.num_relations = num_relations

        self.W_Q = nn.ModuleList([
            nn.Linear(in_channels, out_channels * heads, bias=False) for _ in range(num_relations)
        ])
        self.W_K = nn.ModuleList([
            nn.Linear(in_channels, out_channels * heads, bias=False) for _ in range(num_relations)
        ])
        self.W_V = nn.ModuleList([
            nn.Linear(in_channels, out_channels * heads, bias=False) for _ in range(num_relations)
        ])


        self.root_lin = nn.Linear(in_channels, out_channels * heads, bias=True)

        self.att_vec = nn.Parameter(torch.Tensor(1, heads, out_channels))
        nn.init.xavier_uniform_(self.att_vec)

        self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        nn.init.zeros_(self.bias)

        self.W_O = nn.Linear(heads * out_channels, out_channels)

    def reset_parameters(self):
        for lin in self.W_Q: lin.reset_parameters()
        for lin in self.W_K: lin.reset_parameters()
        for lin in self.W_V: lin.reset_parameters()
        self.root_lin.reset_parameters()
        self.W_O.reset_parameters()
        nn.init.xavier_uniform_(self.att_vec)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_type):

        total_out = torch.zeros(x.size(0), self.heads * self.out_channels, device=x.device)

        for r in range(self.num_relations):
            mask = (edge_type == r)
            if not mask.any():
                continue

            sub_edge_index = edge_index[:, mask]

            h_q = self.W_Q[r](x).view(-1, self.heads, self.out_channels)
            h_k = self.W_K[r](x).view(-1, self.heads, self.out_channels)
            h_v = self.W_V[r](x).view(-1, self.heads, self.out_channels)

            relation_out = self.propagate(sub_edge_index, x=(h_v, h_q), k=h_k)
            total_out += relation_out.view(-1, self.heads * self.out_channels)

        root_feature = self.root_lin(x)
        total_out = total_out + root_feature + self.bias

        #投影
        return self.W_O(total_out)

    def message(self, x_j, x_i, k_j, index, ptr, size_i):


        score = (x_i * k_j).sum(dim=-1) / (self.out_channels ** 0.5)

        alpha = softmax(score, index, ptr, size_i)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.unsqueeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.5):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x
        h = self.norm1(x)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.lin1(h)
        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.lin2(h)
        return residual + h


class RAVQ4DD(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 local_layers=3, dropout=0.6, heads=4,
                 num_codes=64, num_node_types=3, num_relations=4,
                 use_dynamic_codebook=True, **kwargs):
        super(RAVQ4DD, self).__init__()

        self.dropout = dropout
        self.use_dynamic_codebook = use_dynamic_codebook
        self.heads = heads
        self.hidden_channels = hidden_channels

        self.type_projector = TypeSpecificProjector(in_channels, hidden_channels * heads, num_node_types)

        self.local_convs = torch.nn.ModuleList()
        self.vqs = torch.nn.ModuleList()

        self.lns = torch.nn.ModuleList()


        for _ in range(local_layers):

            self.local_convs.append(
                RelationAwareLayer(hidden_channels * heads, hidden_channels,
                                   num_relations=num_relations, heads=heads, dropout=dropout)
            )


            self.lins_restore = torch.nn.ModuleList([
                torch.nn.Linear(hidden_channels, hidden_channels * heads) for _ in range(local_layers)
            ])

            self.vqs.append(
                DynamicResidualVectorQuant(
                    dim=hidden_channels,
                    codebook_size=num_codes,
                    num_res_layers=3,
                    use_dynamic_generator=use_dynamic_codebook,
                    num_node_types=num_node_types
                )
            )

            self.lns.append(torch.nn.LayerNorm(hidden_channels * heads))

        vq_concat_dim = local_layers * hidden_channels


        self.id_classifier = nn.Sequential(
            nn.Linear(vq_concat_dim, hidden_channels * 2),
            ResidualBlock(hidden_channels * 2, dropout),
            nn.LayerNorm(hidden_channels * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, out_channels)
        )

        self.aux_path_predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

        self.pred_local = torch.nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        self.type_projector.reset_parameters()
        for conv in self.local_convs: conv.reset_parameters()
        for lin in self.lins_restore: lin.reset_parameters()
        for ln in self.lns: ln.reset_parameters()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: m.bias.data.fill_(0.01)

        self.id_classifier.apply(init_weights)
        self.aux_path_predictor.apply(init_weights)
        self.pred_local.reset_parameters()

    def forward(self, x, edge_index, edge_type, node_type, vq_noise_scale=0.0):

        x = self.type_projector(x, node_type)
        x = F.dropout(x, p=self.dropout, training=self.training)

        quantized_list = []
        total_commit_loss = 0
        x_local = 0
        all_indices = []

        curr_x = x

        for i, (conv, vq) in enumerate(zip(self.local_convs, self.vqs)):

            x_norm = self.lns[i](curr_x)

            h_conv = conv(x_norm, edge_index, edge_type)
            h_conv = F.gelu(h_conv)
            h_conv = F.dropout(h_conv, p=self.dropout, training=self.training)

            x_local = x_local + h_conv

            quantized, layer_indices_list, layer_loss, next_history = vq(
                h_conv,
                node_type=node_type,
                original_feat=h_conv,
                history_context=None,
                training=self.training,
                noise_scale=vq_noise_scale
            )

            total_commit_loss += layer_loss
            quantized_list.append(quantized)

            if len(layer_indices_list) > 0:
                all_indices.append(torch.stack(layer_indices_list, dim=1))


            h_restored = self.lins_restore[i](h_conv)

            curr_x = curr_x + h_restored

        id_concat = torch.cat(quantized_list, dim=1)
        id_out = self.id_classifier(id_concat)

        aux_out = self.aux_path_predictor(h_conv)
        gnn_out = self.pred_local(x_local)

        if len(all_indices) > 0:
            final_indices = torch.cat(all_indices, dim=1)
        else:
            final_indices = None

        return id_out, aux_out, gnn_out, total_commit_loss, final_indices