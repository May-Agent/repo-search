"""The traced torch module: EmbeddingGemma 2 text encoder at one fixed length.

Attention is the eager path — explicit matmul + mask + softmax + matmul — because the
fused SDPA op does not compile for the Neural Engine. RoPE cos/sin tables and the
sliding-window band are registered as constant buffers so they bake into the model.
Output is mean-pooled over the attention mask, amax pre-scaled, then L2-normalized.
"""

import torch
from transformers.masking_utils import create_bidirectional_sliding_window_mask

NEG = -1e4


class Encoder(torch.nn.Module):
    def __init__(self, lm, length):
        super().__init__()
        self.lm = lm
        cfg = lm.config
        pos = torch.arange(length).unsqueeze(0)
        dummy = torch.zeros(1, length, cfg.hidden_size)
        for t in set(cfg.layer_types):
            cos, sin = lm.rotary_emb(dummy, pos, t)
            self.register_buffer(f"cos_{t}", cos)
            self.register_buffer(f"sin_{t}", sin)
        band = create_bidirectional_sliding_window_mask(
            config=cfg,
            inputs_embeds=dummy,
            attention_mask=torch.ones(1, length, dtype=torch.long),
        )
        if band is None:
            band = torch.zeros(1, 1, length, length)
        elif band.dtype == torch.bool:
            band = torch.where(band, 0.0, NEG).view(1, 1, length, length)
        else:
            band = torch.where(band < -1, NEG, 0.0).view(1, 1, length, length)
        self.register_buffer("band", band.float())

        self.length = length

    def attention(self, i, a, h, mask, cos, sin):
        cfg, L = self.lm.config, self.length
        nh, hd = cfg.num_attention_heads, a.head_dim
        nkv = cfg.per_layer_config[i].num_key_value_heads
        q = a.q_norm(a.q_proj(h).view(1, L, nh, hd).transpose(1, 2))
        k = a.k_norm(a.k_proj(h).view(1, L, nkv, hd).transpose(1, 2))
        v = a.v_norm(a.v_proj(h).view(1, L, nkv, hd).transpose(1, 2))
        cos, sin, half = cos.unsqueeze(1), sin.unsqueeze(1), hd // 2
        q = q * cos + torch.cat((-q[..., half:], q[..., :half]), -1) * sin
        k = k * cos + torch.cat((-k[..., half:], k[..., :half]), -1) * sin
        if nkv != nh:
            k = k[:, :, None].expand(1, nkv, nh // nkv, L, hd).reshape(1, nh, L, hd)
            v = v[:, :, None].expand(1, nkv, nh // nkv, L, hd).reshape(1, nh, L, hd)
        w = torch.softmax(torch.matmul(q, k.transpose(2, 3)) * a.scaling + mask, dim=-1)
        o = torch.matmul(w, v).transpose(1, 2).reshape(1, L, nh * hd)
        return a.o_proj(o)

    def forward(self, input_ids, attention_mask):
        lm, cfg, L = self.lm, self.lm.config, self.length
        x = lm.embed_tokens(input_ids)
        p = lm.ple
        ple = p.per_layer_projection_norm(
            (p.per_layer_model_projection(x) * p.per_layer_model_projection_scale)
            .view(1, L, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input))
        pad = ((1.0 - attention_mask) * NEG).view(1, 1, 1, L)
        masks = {"full_attention": pad, "sliding_attention": pad + self.band}
        h = x
        for i, layer in enumerate(lm.layers):
            t = cfg.layer_types[i]
            r = h
            h = layer.post_attention_layernorm(self.attention(
                i, layer.self_attn, layer.input_layernorm(h), masks[t],
                getattr(self, f"cos_{t}"), getattr(self, f"sin_{t}")))
            h = r + h
            r = h
            h = r + layer.post_feedforward_layernorm(layer.mlp(layer.pre_feedforward_layernorm(h)))
            b = layer.ple_block
            g = b.act_fn(b.per_layer_input_gate(h)) * ple[:, :, i, :]
            h = h + b.post_per_layer_input_norm(b.per_layer_projection(g))
            h = h * layer.layer_scalar
        h = lm.norm(h)
        h = lm.embedding_projection(h)
        m = attention_mask.unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1)
        pooled = pooled / pooled.abs().amax(-1, keepdim=True)
        return pooled / torch.sqrt((pooled * pooled).sum(-1, keepdim=True))
