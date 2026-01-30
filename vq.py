import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicEuclideanCodebook(nn.Module):
    def __init__(self, dim, codebook_size, num_codebooks=1, decay=0.95, eps=1e-5,
                 use_dynamic_generator=False, context_dim=None, num_node_types=10, chunk_size=None):
        super().__init__()
        self.decay = decay
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.dim = dim
        self.use_dynamic_generator = use_dynamic_generator

        assert dim % num_codebooks == 0, f"dim {dim} must be divisible by num_codebooks {num_codebooks}"
        self.head_dim = dim // num_codebooks

        init_embed = torch.randn(num_codebooks, codebook_size, self.head_dim)
        self.register_buffer('embed', init_embed)
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))
        self.register_buffer('embed_avg', init_embed.clone())

        self.type_embedding = nn.Embedding(num_node_types, dim // 2)

        if self.use_dynamic_generator:
            assert context_dim is not None
            input_dim = context_dim + (dim // 2)
            self.generator_mlp = nn.Sequential(
                nn.Linear(input_dim, dim),
                nn.LeakyReLU(),
                nn.Linear(dim, num_codebooks * codebook_size * self.head_dim)
            )
            nn.init.zeros_(self.generator_mlp[-1].weight)
            nn.init.zeros_(self.generator_mlp[-1].bias)

    def forward(self, x, node_type, context=None):
        N, D = x.shape
        x_reshaped = x.view(N, self.num_codebooks, self.head_dim)
        effective_embed = self.embed.clone()

        if self.use_dynamic_generator and context is not None:
            type_emb = self.type_embedding(node_type)
            full_ctx = torch.cat([context, type_emb], dim=-1)
            delta = self.generator_mlp(full_ctx)
            delta = delta.view(N, self.num_codebooks, self.codebook_size, self.head_dim)
            effective_embed = effective_embed.unsqueeze(0) + delta
        else:
            effective_embed = effective_embed.unsqueeze(0)

        x_exp = x_reshaped.unsqueeze(2)
        dist = torch.sum((x_exp - effective_embed) ** 2, dim=-1)
        embed_ind = dist.argmin(dim=-1)

        ind_expanded = embed_ind.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.head_dim)
        if effective_embed.shape[0] == 1:
            effective_embed = effective_embed.expand(N, -1, -1, -1)

        quantized = torch.gather(effective_embed, 2, ind_expanded).squeeze(2)
        quantized_out = quantized.view(N, D)

        if self.training:
            flat_ind = embed_ind.view(-1)
            one_hot = F.one_hot(flat_ind, self.codebook_size).type(x.dtype)
            one_hot_reshaped = one_hot.view(N, self.num_codebooks, self.codebook_size)
            current_cluster_size = one_hot_reshaped.sum(dim=0)
            current_embed_sum = (x_reshaped.unsqueeze(2) * one_hot_reshaped.unsqueeze(3)).sum(dim=0)

            self.cluster_size.data.lerp_(current_cluster_size, 1 - self.decay)
            self.embed_avg.data.lerp_(current_embed_sum, 1 - self.decay)
            n = self.cluster_size.sum(dim=-1, keepdim=True)
            cluster_size = (self.cluster_size + 1e-5) / (n + self.codebook_size * 1e-5) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(-1)
            self.embed.data.copy_(embed_normalized)

        return quantized_out, embed_ind, 0


class DynamicResidualVectorQuant(nn.Module):
    def __init__(self, dim, codebook_size, num_res_layers=3, use_dynamic_generator=False, num_node_types=10, **kwargs):
        super().__init__()
        self.num_res_layers = num_res_layers
        self.layers = nn.ModuleList()
        context_dim = dim * 2

        for _ in range(num_res_layers):
            self.layers.append(
                DynamicEuclideanCodebook(
                    dim=dim,
                    codebook_size=codebook_size,
                    use_dynamic_generator=use_dynamic_generator,
                    context_dim=context_dim,
                    num_node_types=num_node_types,
                    decay=0.95
                )
            )

    def forward(self, x, node_type, original_feat=None, history_context=None, training=True, noise_scale=0.0):
        curr_x = x
        if training and noise_scale > 0:
            noise = torch.randn_like(curr_x) * noise_scale
            curr_x = curr_x + noise

        quantized_outputs = []
        embed_inds = []
        total_loss = 0

        for layer in self.layers:
            if original_feat is not None and history_context is not None:
                ctx = torch.cat([original_feat, history_context], dim=-1)
            else:
                ctx = None
            quantized, embed_ind, _ = layer(curr_x, node_type, context=ctx)
            curr_x = curr_x - quantized
            quantized_outputs.append(quantized)
            embed_inds.append(embed_ind)
            total_loss = total_loss + F.mse_loss(quantized, x.detach())
            if history_context is not None:
                history_context = history_context + quantized.detach()

        final_quantized = sum(quantized_outputs)
        return final_quantized, embed_inds, total_loss, history_context