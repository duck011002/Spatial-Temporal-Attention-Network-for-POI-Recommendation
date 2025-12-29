from layers import *


class Model(nn.Module):
    def __init__(self, t_dim, l_dim, u_dim, embed_dim, ex, dropout=0.1):
        super(Model, self).__init__()
        emb_t = nn.Embedding(t_dim, embed_dim, padding_idx=0)
        emb_l = nn.Embedding(l_dim, embed_dim, padding_idx=0)
        emb_u = nn.Embedding(u_dim, embed_dim, padding_idx=0)
        emb_su = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_sl = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_tu = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_tl = nn.Embedding(2, embed_dim, padding_idx=0)
        embed_layers = emb_t, emb_l, emb_u, emb_su, emb_sl, emb_tu, emb_tl

        self.MultiEmbed = MultiEmbed(ex, embed_dim, embed_layers)
        self.SelfAttn = SelfAttn(embed_dim, embed_dim)
        self.Embed = Embed(ex, embed_dim, l_dim-1, embed_layers)
        self.Attn = Attn(emb_l, l_dim-1)

    def forward(self, traj, mat1, poi_coords, vec, traj_len, cand_locs):
        # long(N, M, [u, l, t]), float(N, M, M, 2), float(L, L), float(N, M), long(N)
        # Updated: poi_coords (L+1, 2) tensor, cand_locs (N, K)
        
        joint, delta = self.MultiEmbed(traj, mat1, traj_len)  # (N, M, emb), (N, M, M, emb)
        self_attn = self.SelfAttn(joint, delta, traj_len)  # (N, M, emb)
        
        # Pass poi_coords and cand_locs to Embed
        self_delta = self.Embed(traj[:, :, 1], poi_coords, vec, traj_len, cand_locs)  # (N, M, K, emb)
        
        # Pass cand_locs to Attn
        output = self.Attn(self_attn, self_delta, traj_len, cand_locs)  # (N, K)
        return output


class ModelHDGMoE(nn.Module):
    def __init__(self, t_dim, l_dim, u_dim, embed_dim, ex, config):
        super(ModelHDGMoE, self).__init__()
        emb_t = nn.Embedding(t_dim, embed_dim, padding_idx=0)
        emb_l = nn.Embedding(l_dim, embed_dim, padding_idx=0)
        emb_u = nn.Embedding(u_dim, embed_dim, padding_idx=0)
        emb_su = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_sl = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_tu = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_tl = nn.Embedding(2, embed_dim, padding_idx=0)
        embed_layers = emb_t, emb_l, emb_u, emb_su, emb_sl, emb_tu, emb_tl

        self.MultiEmbed = MultiEmbed(ex, embed_dim, embed_layers)
        
        # New Encoder Stack
        self.Encoder = EncoderStack(
            embed_dim, 
            config['model']['num_layers'], 
            config['model']['num_heads'], 
            config
        )
        
        # Keep existing prediction layers
        self.Embed = Embed(ex, embed_dim, l_dim-1, embed_layers)
        self.Attn = Attn(emb_l, l_dim-1)
        
        self.config = config
        self.emb_u = emb_u # Needed for gate2 user features
        
        # Optional Cat/Admin embeddings
        if config['data']['use_category']:
            cat_dim = config['data']['category_vocab_size']
            self.emb_cat = nn.Embedding(cat_dim, embed_dim, padding_idx=0)
        
        if config['data']['use_admin']:
            admin_dim = config['data']['admin_vocab_size']
            self.emb_admin = nn.Embedding(admin_dim, embed_dim, padding_idx=0)

    def forward(self, traj, mat1, poi_coords, mat2t, traj_len, cand_locs):
        # 1. MultiEmbed
        joint, delta = self.MultiEmbed(traj, mat1, traj_len)
        
        # 2. Gate Features Preparation
        N, M, D = joint.shape
        
        # Gate 1 Features
        # mat1 (N, M, M, 2) -> diagonal offset -1
        mat1_perm = mat1.permute(0, 3, 1, 2) # (N, 2, M, M)
        diag_adj = torch.diagonal(mat1_perm, offset=-1, dim1=-2, dim2=-1) # (N, 2, M-1)
        adj_pad = torch.zeros((N, 2, 1), device=traj.device)
        adj_vals = torch.cat([adj_pad, diag_adj], dim=-1).permute(0, 2, 1) # (N, M, 2)
        
        delta_s_adj = adj_vals[:, :, 0].unsqueeze(-1)
        delta_t_adj = adj_vals[:, :, 1].unsqueeze(-1)
        eps = 1e-6
        v_adj = delta_s_adj / (delta_t_adj + eps)
        
        gate1_adj = torch.cat([delta_s_adj, delta_t_adj, v_adj], dim=-1) # (N, M, 3)
        
        time_ids = traj[:, :, 2].long()
        gate1_time = self.MultiEmbed.emb_t(time_ids) 
        
        gate1_feats = {'adj': gate1_adj, 'time': gate1_time}
        
        # Gate 2 Features
        user_ids = traj[:, :, 0].long()
        user_emb = self.emb_u(user_ids)
        gate2_feats = {'user': user_emb}
        
        if self.config['data']['use_category'] and traj.shape[-1] > 3:
             cat_ids = traj[:, :, 3].long()
             gate2_feats['cat'] = self.emb_cat(cat_ids)
             
        if self.config['data']['use_admin'] and traj.shape[-1] > 4:
             admin_ids = traj[:, :, 4].long()
             gate2_feats['admin'] = self.emb_admin(admin_ids)
             
        # 3. Encoder Stack
        attn_bias = None
        if self.config['model']['attn_bias_mode'] == 'sum':
            attn_bias = delta.sum(-1) # (N, M, M)
            
        idx = torch.arange(M, device=traj.device)[None, :]
        pad_mask = (idx < traj_len[:, None]).float() # (N, M)
        
        encoder_out, aux = self.Encoder(joint, attn_bias, pad_mask, gate1_feats, gate2_feats)
        
        # 4. Final Prediction
        self_delta = self.Embed(traj[:, :, 1], poi_coords, mat2t, traj_len, cand_locs)
        score = self.Attn(encoder_out, self_delta, traj_len, cand_locs)
        
        return score, aux
