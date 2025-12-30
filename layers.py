from load import *
import torch
from torch import nn
from torch.nn import functional as F
import math

seed = 0
global_seed = 0
hours = 24 * 7
torch.manual_seed(seed)
device = 'cuda'


def to_npy(x):
    return x.cpu().data.numpy() if device == 'cuda' else x.detach().numpy()


class Attn(nn.Module):
    def __init__(self, emb_loc, loc_max, dropout=0.1):
        super(Attn, self).__init__()
        self.value = nn.Linear(max_len, 1, bias=False)
        self.emb_loc = emb_loc
        self.loc_max = loc_max

    def forward(self, self_attn, self_delta, traj_len, cand_locs=None):
        # self_attn (N, M, emb), self_delta (N, M, K, emb), cand_locs (N, K)
        # candidates = torch.linspace(1, int(self.loc_max), int(self.loc_max)).long()  # (L)
        # candidates = candidates.unsqueeze(0).expand(N, -1).to(device)  # (N, L)

        candidates = cand_locs  # (N, K)
        self_delta = torch.sum(self_delta, -1).transpose(-1, -2)  # squeeze the embed dimension -> (N, K, M)
        # [N, K, M] = self_delta.shape

        emb_candidates = self.emb_loc(candidates)  # (N, K, emb)
        attn = torch.mul(torch.bmm(emb_candidates, self_attn.transpose(-1, -2)), self_delta)  # (N, K, M)
        # pdb.set_trace()
        attn_out = self.value(attn).squeeze(-1)  # (N, K)
        # attn_out = F.log_softmax(attn_out, dim=-1)  # ignore if cross_entropy_loss

        return attn_out  # (N, K)


class SelfAttn(nn.Module):
    def __init__(self, emb_size, output_size, dropout=0.1):
        super(SelfAttn, self).__init__()
        self.query = nn.Linear(emb_size, output_size, bias=False)
        self.key = nn.Linear(emb_size, output_size, bias=False)
        self.value = nn.Linear(emb_size, output_size, bias=False)

    def forward(self, joint, delta, traj_len):
        delta = torch.sum(delta, -1)  # squeeze the embed dimension
        # joint (N, M, emb), delta (N, M, M, emb), len [N]
        # construct attention mask

        # Vectorized mask
        # mask = torch.zeros_like(delta, dtype=torch.float32)
        # for i in range(mask.shape[0]):
        #     mask[i, 0:traj_len[i], 0:traj_len[i]] = 1

        N, M, _ = joint.shape
        idx = torch.arange(M, device=joint.device)[None, :]
        valid = (idx < traj_len[:, None])  # (N, M)
        mask = (valid[:, :, None] & valid[:, None, :]).float()  # (N, M, M)

        attn = torch.add(torch.bmm(self.query(joint), self.key(joint).transpose(-1, -2)), delta)  # (N, M, M)
        attn = F.softmax(attn, dim=-1) * mask  # (N, M, M)

        attn_out = torch.bmm(attn, self.value(joint))  # (N, M, emb)

        return attn_out  # (N, M, emb)


class Embed(nn.Module):
    def __init__(self, ex, emb_size, loc_max, embed_layers):
        super(Embed, self).__init__()
        _, _, _, self.emb_su, self.emb_sl, self.emb_tu, self.emb_tl = embed_layers
        self.su, self.sl, self.tu, self.tl = ex
        self.emb_size = emb_size
        self.loc_max = loc_max

    def forward(self, traj_loc, poi_coords, vec, traj_len, cand_locs):
        # traj_loc (N, M), mat2 (L, L), vec (N, M), delta_t (N, M, L)
        # Now: traj_loc (N, M), poi_coords (L+1, 2), vec (N, M), cand_locs (N, K)

        N, M = traj_loc.shape
        K = cand_locs.shape[1]

        delta_t = vec.unsqueeze(-1).expand(-1, -1, K)  # (N, M, K)

        # Calculate delta_s using Haversine
        # Need to gather coords
        traj_xy = poi_coords[traj_loc]  # (N, M, 2)
        cand_xy = poi_coords[cand_locs]  # (N, K, 2)

        # Broadcasting for (N, M, K)
        # lat1, lon1 from traj_xy (N, M, 1, 2)
        # lat2, lon2 from cand_xy (N, 1, K, 2)

        lat1 = traj_xy[:, :, 1].unsqueeze(2) * math.pi / 180.0
        lon1 = traj_xy[:, :, 0].unsqueeze(
            2) * math.pi / 180.0  # Note: poi_coords was poi[:, 1:3] -> [lat, lon]. Wait, original load.py haversine uses lon=1, lat=2??
        # load.py: mat[i, j, 0] = haversine(lon1=poi_item[2], lat1=poi_item[1] ...
        # poi_item is row from poi. poi is (L, [l, lat, lon]).
        # so index 1 is lat, index 2 is lon.
        # poi_coords = poi[:, 1:3] -> col 0 is lat, col 1 is lon.
        # so traj_xy[..., 0] is lat, traj_xy[..., 1] is lon.

        lat1 = traj_xy[:, :, 0].unsqueeze(2) * math.pi / 180.0
        lon1 = traj_xy[:, :, 1].unsqueeze(2) * math.pi / 180.0
        lat2 = cand_xy[:, :, 0].unsqueeze(1) * math.pi / 180.0
        lon2 = cand_xy[:, :, 1].unsqueeze(1) * math.pi / 180.0

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))
        R = 6371.0
        delta_s = c * R  # (N, M, K)

        # vectorized mask
        idx = torch.arange(M, device=traj_loc.device)[None, :]
        mask = (idx < traj_len[:, None]).long()  # (N, M)

        esl, esu, etl, etu = self.emb_sl(mask), self.emb_su(mask), self.emb_tl(mask), self.emb_tu(mask)
        # These are (N, M, emb)
        # Expand to (N, M, K, emb)
        esl = esl.unsqueeze(2).expand(-1, -1, K, -1)
        esu = esu.unsqueeze(2).expand(-1, -1, K, -1)
        etl = etl.unsqueeze(2).expand(-1, -1, K, -1)
        etu = etu.unsqueeze(2).expand(-1, -1, K, -1)

        vsl, vsu, vtl, vtu = (delta_s - self.sl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
            (self.su - delta_s).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
            (delta_t - self.tl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
            (self.tu - delta_t).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)

        space_interval = (esl * vsu + esu * vsl) / (self.su - self.sl)
        time_interval = (etl * vtu + etu * vtl) / (self.tu - self.tl)
        delta = space_interval + time_interval  # (N, M, K, emb)

        return delta


class MultiEmbed(nn.Module):
    def __init__(self, ex, emb_size, embed_layers):
        super(MultiEmbed, self).__init__()
        self.emb_t, self.emb_l, self.emb_u, \
            self.emb_su, self.emb_sl, self.emb_tu, self.emb_tl = embed_layers
        self.su, self.sl, self.tu, self.tl = ex
        self.emb_size = emb_size

    def forward(self, traj, mat, traj_len):
        # traj (N, M, 3), mat (N, M, M, 2), len [N]
        traj[:, :, 2] = (traj[:, :, 2] - 1) % hours + 1  # segment time by 24 hours * 7 days
        time = self.emb_t(traj[:, :, 2])  # (N, M) --> (N, M, embed)
        loc = self.emb_l(traj[:, :, 1])  # (N, M) --> (N, M, embed)
        user = self.emb_u(traj[:, :, 0])  # (N, M) --> (N, M, embed)
        joint = time + loc + user  # (N, M, embed)

        delta_s, delta_t = mat[:, :, :, 0], mat[:, :, :, 1]  # (N, M, M)

        # Vectorized mask
        N, M, _ = delta_s.shape
        idx = torch.arange(M, device=traj.device)[None, :]
        mask_bool = (idx < traj_len[:, None])[:, :, None] & (idx < traj_len[:, None])[:, None, :]
        mask = mask_bool.long()  # (N, M, M)

        esl, esu, etl, etu = self.emb_sl(mask), self.emb_su(mask), self.emb_tl(mask), self.emb_tu(mask)
        vsl, vsu, vtl, vtu = (delta_s - self.sl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
            (self.su - delta_s).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
            (delta_t - self.tl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
            (self.tu - delta_t).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)

        space_interval = (esl * vsu + esu * vsl) / (self.su - self.sl)
        time_interval = (etl * vtu + etu * vtl) / (self.tu - self.tl)
        delta = space_interval + time_interval  # (N, M, M, emb)

        return joint, delta


# -----------------------------------------------------
# HDG-MoE Components
# -----------------------------------------------------

class MultiHeadSelfAttentionWithBias(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttentionWithBias, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_bias=None, key_padding_mask=None):
        # x: (N, L, D)
        # attn_bias: (N, L, L) or (N, H, L, L) - additive bias (0 or -inf)
        # key_padding_mask: (N, L) - boolean, True means VALID, False means PADDING (aligned with existing STAN logic)

        N, L, D = x.shape

        q = self.q_proj(x).view(N, L, self.num_heads, self.head_dim).transpose(1, 2)  # (N, H, L, d)
        k = self.k_proj(x).view(N, L, self.num_heads, self.head_dim).transpose(1, 2)  # (N, H, L, d)
        v = self.v_proj(x).view(N, L, self.num_heads, self.head_dim).transpose(1, 2)  # (N, H, L, d)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (N, H, L, L)

        # Apply attn_bias
        if attn_bias is not None:
            # attn_bias could be (N, L, L). Expand to (N, 1, L, L) broadcase
            if attn_bias.dim() == 3:
                attn_bias = attn_bias.unsqueeze(1)
            scores = scores + attn_bias

        # Apply padding mask (set to -inf)
        # key_padding_mask: (N, L). True=Valid.
        if key_padding_mask is not None:
            # We want to mask POSITIONS where padding is.
            # Convert True(Valid) to False(Mask), False(Pad) to True(Mask) for pytorch conventions?
            # Current STAN logic: mask[i, :len] = 1 (True). So True is Valid.
            # We want to mask keys.
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (N, 1, 1, L)
            # mask has 1 for valid, 0 for invalid.
            # We want to add -inf where mask is 0.
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)  # (N, H, L, d)
        output = output.transpose(1, 2).contiguous().view(N, L, D)
        output = self.out_proj(output)
        return output


class TwoStageGatedMoE(nn.Module):
    def __init__(self, embed_dim, config):
        super(TwoStageGatedMoE, self).__init__()
        self.embed_dim = embed_dim
        self.config = config

        # Expert Configuration
        self.groups = config['mechanism_groups']
        self.experts_per_group = config['experts_per_group']  # list
        self.total_experts = sum(self.experts_per_group)
        self.hidden_dim = int(embed_dim * config['expert_hidden_mult'])

        # Experts ModuleList (original structure for compatibility)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, embed_dim)
            ) for _ in range(self.total_experts)
        ])

        # Gate 1 (Mechanism)
        gate1_input_dim = embed_dim + 3 + embed_dim
        self.gate1 = nn.Linear(gate1_input_dim, self.groups)

        # Gate 2 (Scenario)
        gate2_input_base = embed_dim + embed_dim
        if config['use_category']: gate2_input_base += embed_dim
        if config['use_admin']: gate2_input_base += embed_dim

        self.gate2_layers = nn.ModuleList([
            nn.Linear(gate2_input_base, k) for k in self.experts_per_group
        ])

        self.k1 = config['gate1_topk']
        self.k2 = config['gate2_topk']
        self.temp = config['gate_temperature']
        self.router_eps = config['router_eps']

    def forward(self, x, gate1_features, gate2_features, padding_mask):
        N, L, D = x.shape
        valid_mask = padding_mask > 0  # Boolean

        # --- Gate 1 Forward ---
        g1_in = torch.cat([x, gate1_features['adj'], gate1_features['time']], dim=-1)
        g1_logits = self.gate1(g1_in) / self.temp
        g1_probs = F.softmax(g1_logits, dim=-1)  # (N, L, G)

        # TopK Gate 1
        g1_topk_vals, g1_topk_indices = torch.topk(g1_probs, self.k1, dim=-1)  # (N, L, k1)

        # --- Gate 2 Forward ---
        g2_inputs_list = [x, gate2_features['user']]
        if 'cat' in gate2_features and gate2_features['cat'] is not None:
            g2_inputs_list.append(gate2_features['cat'])
        if 'admin' in gate2_features and gate2_features['admin'] is not None:
            g2_inputs_list.append(gate2_features['admin'])
        g2_in = torch.cat(g2_inputs_list, dim=-1)

        final_output = torch.zeros_like(x)

        # Aux stats
        lb1 = 0.0
        lb2 = 0.0
        gate1_usage = torch.zeros(self.groups, device=x.device)
        gate2_usage = [torch.zeros(k, device=x.device) for k in self.experts_per_group]

        # LB1 Calculation (vectorized)
        if self.config['ignore_padding_in_lb'] and valid_mask.any():
            p1_mean = g1_probs[valid_mask].mean(0)
            lb1 = torch.sum((p1_mean - 1.0 / self.groups) ** 2)
        else:
            p1_mean = g1_probs.mean(0).mean(0)
            lb1 = torch.sum((p1_mean - 1.0 / self.groups) ** 2)

        # Gate 1 Usage (vectorized - faster than loop)
        with torch.no_grad():
            if valid_mask.any():
                flat_g1_idx = g1_topk_indices[valid_mask].flatten()
                gate1_usage.scatter_add_(0, flat_g1_idx, torch.ones_like(flat_g1_idx, dtype=gate1_usage.dtype))

        # Process each group
        expert_offset = 0
        for g_idx in range(self.groups):
            num_sub_experts = self.experts_per_group[g_idx]

            # Gate 2 for this group
            g2_logits_group = self.gate2_layers[g_idx](g2_in)
            g2_probs_group = F.softmax(g2_logits_group, dim=-1)  # (N, L, K_m)

            # LB2 Calculation
            is_selected = (g1_topk_indices == g_idx).any(dim=-1)  # (N, L)
            is_selected_valid = is_selected & valid_mask

            if self.config['ignore_padding_in_lb'] and is_selected_valid.any():
                p2_mean = g2_probs_group[is_selected_valid].mean(0)
                lb2 += torch.sum((p2_mean - 1.0 / num_sub_experts) ** 2)

            # Top-1 from Gate 2
            g2_vals, g2_indices = torch.topk(g2_probs_group, self.k2, dim=-1)  # (N, L, 1)

            # Gate 2 Usage (vectorized)
            with torch.no_grad():
                if is_selected_valid.any():
                    flat_g2_idx = g2_indices[is_selected_valid].flatten()
                    gate2_usage[g_idx].scatter_add_(0, flat_g2_idx,
                                                    torch.ones_like(flat_g2_idx, dtype=gate2_usage[g_idx].dtype))

            # Compute weights: w1 for this group
            mask_pos = (g1_topk_indices == g_idx)
            w1_group = (g1_topk_vals * mask_pos.float()).sum(dim=-1, keepdim=True)  # (N, L, 1)

            group_output = torch.zeros_like(x)

            # Expert computation - gather all active tokens for this group, process per expert
            for k in range(num_sub_experts):
                mask_k = (g2_indices == k) & (w1_group > 0)  # (N, L, 1)
                mask_k = mask_k.squeeze(-1)

                if mask_k.any():
                    batch_idx = mask_k.nonzero(as_tuple=True)
                    if batch_idx[0].numel() > 0:
                        inp = x[batch_idx]  # (Num_Selected, D)
                        out = self.experts[expert_offset + k](inp)  # (Num_Selected, D)

                        w = w1_group[batch_idx] * g2_vals[batch_idx].squeeze(-1).unsqueeze(1)  # (Num_Selected, 1)
                        weighted_out = out * w

                        group_output[batch_idx] += weighted_out

            final_output += group_output
            expert_offset += num_sub_experts

        aux = {
            'lb1': lb1,
            'lb2': lb2,
            'gate1_usage': gate1_usage,
            'gate2_usage': gate2_usage
        }
        return final_output, aux

        aux = {
            'lb1': lb1,
            'lb2': lb2,
            'gate1_usage': gate1_usage,
            'gate2_usage': gate2_usage
        }
        return final_output, aux


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, config, model_config):
        super(EncoderLayer, self).__init__()
        self.prenorm = model_config.get('prenorm', True)
        self.mha = MultiHeadSelfAttentionWithBias(embed_dim, num_heads, model_config['dropout'])
        self.moe = TwoStageGatedMoE(embed_dim, config)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(model_config['dropout'])
        self.dropout2 = nn.Dropout(model_config['dropout'])

    def forward(self, x, attn_bias, pad_mask, gate1_feats, gate2_feats):
        # x: (N, L, D)
        # PreNorm structure

        # MHA Block
        res = x
        if self.prenorm:
            x_norm = self.norm1(x)
            mha_out = self.mha(x_norm, attn_bias=attn_bias, key_padding_mask=pad_mask)
            x = res + self.dropout1(mha_out)
        else:
            mha_out = self.mha(x, attn_bias=attn_bias, key_padding_mask=pad_mask)
            x = self.norm1(res + self.dropout1(mha_out))

        # MoE Block
        res = x
        if self.prenorm:
            x_norm = self.norm2(x)
            moe_out, aux = self.moe(x_norm, gate1_feats, gate2_feats, pad_mask)
            x = res + self.dropout2(moe_out)
        else:
            moe_out, aux = self.moe(x, gate1_feats, gate2_feats, pad_mask)
            x = self.norm2(res + self.dropout2(moe_out))

        return x, aux


class EncoderStack(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, config):
        super(EncoderStack, self).__init__()

        # Merge configs for layers that need multiple sections
        layer_config = config['moe'].copy()
        if 'data' in config:
            layer_config.update(config['data'])
        if 'loss' in config:
            layer_config.update(config['loss'])

        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, layer_config, config['model'])
            for _ in range(num_layers)
        ])

    def forward(self, x, attn_bias, pad_mask, gate1_feats, gate2_feats):
        total_aux = {'lb1': 0.0, 'lb2': 0.0}
        total_gate1_usage = None
        total_gate2_usage = None

        for layer in self.layers:
            x, aux = layer(x, attn_bias, pad_mask, gate1_feats, gate2_feats)

            total_aux['lb1'] += aux['lb1']
            total_aux['lb2'] += aux['lb2']

            if total_gate1_usage is None:
                total_gate1_usage = aux['gate1_usage']
                total_gate2_usage = aux['gate2_usage']
            else:
                total_gate1_usage += aux['gate1_usage']
                for i in range(len(total_gate2_usage)):
                    total_gate2_usage[i] += aux['gate2_usage'][i]

        total_aux['gate1_usage'] = total_gate1_usage
        total_aux['gate2_usage'] = total_gate2_usage

        return x, total_aux
