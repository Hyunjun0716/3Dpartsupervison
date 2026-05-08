import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np

from .common import *
from .attention import CrossAttention


class FiLM(Module):
    """
    Feature-wise Linear Modulation for global text conditioning
    """
    def __init__(self, text_dim, feature_dim):
        super().__init__()
        self.gamma_layer = torch.nn.Linear(text_dim, feature_dim)
        self.beta_layer = torch.nn.Linear(text_dim, feature_dim)

    def forward(self, features, text_pool):
        """
        Args:
            features: (B, N, feature_dim) - point features
            text_pool: (B, text_dim) - pooled text embedding
        Returns:
            modulated_features: (B, N, feature_dim)
        """
        gamma = self.gamma_layer(text_pool).unsqueeze(1)  # (B, 1, feature_dim)
        beta = self.beta_layer(text_pool).unsqueeze(1)    # (B, 1, feature_dim)
        return gamma * features + beta


class VarianceSchedule(Module):

    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


class PointwiseNet(Module):

    def __init__(self, point_dim, context_dim, residual, text_dim=0, use_cross_attention=True, tokenizer=None):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.text_dim = text_dim
        self.use_cross_attention = use_cross_attention and text_dim > 0
        self.use_film = use_cross_attention and text_dim > 0  # Enable FiLM when text is available
        self.tokenizer = tokenizer  # For stop word filtering

        # Context dimension: time (3) + z_latent (context_dim)
        # Text is handled separately via FiLM and cross-attention to save memory
        total_ctx_dim = 3 + context_dim

        # 6-layer MLP (memory-efficient while maintaining capacity)
        self.layers = ModuleList([
            ConcatSquashLinear(point_dim, 128, total_ctx_dim),
            ConcatSquashLinear(128, 256, total_ctx_dim),
            ConcatSquashLinear(256, 512, total_ctx_dim),
            ConcatSquashLinear(512, 256, total_ctx_dim),
            ConcatSquashLinear(256, 128, total_ctx_dim),
            ConcatSquashLinear(128, point_dim, total_ctx_dim)
        ])

        # Latent code projection for more expressiveness
        self.z_projection = nn.Sequential(
            nn.Linear(context_dim, context_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(context_dim * 2, context_dim)
        )

        # Multi-scale Cross Attention
        # Apply attention at different feature dimensions for hierarchical text-shape alignment
        if self.use_cross_attention:
            # 256-dim: Mid-level features (part structures like armrests, legs)
            self.cross_attn_256 = CrossAttention(
                query_dim=256,
                context_dim=text_dim,
                heads=4,  # Fewer heads for lower dimension
                dim_head=64,
                dropout=0.0,
                use_stopword_filter=True,
                stopword_penalty=0.1
            )
            self.cross_attn_norm_256 = nn.LayerNorm(256)

            # 512-dim: High-level features (global semantics like category(chair), style)
            self.cross_attn_512 = CrossAttention(
                query_dim=512,
                context_dim=text_dim,
                heads=8,  # More heads for higher dimension
                dim_head=64,
                dropout=0.0,
                use_stopword_filter=True,
                stopword_penalty=0.1
            )
            self.cross_attn_norm_512 = nn.LayerNorm(512)

        # FiLM modulation at 512-dim layers (layer 2 and 3)
        if self.use_film:
            self.film_1 = FiLM(text_dim, 512)
            self.film_2 = FiLM(text_dim, 512)


    def forward(self, x, beta, context, text_emb=None, return_attention=False):
        """
        Forward pass with multi-scale cross-attention

        Args:
            x: Noisy points (B, N, 3)
            beta: Diffusion timestep (B,)
            context: Shape latents from PointNet (B, z_dim)
            text_emb: Text embeddings from CLIP (dict with 'tokens', 'pool', 'token_ids')
            return_attention: If True, return attention weights

        Returns:
            Predicted noise (B, N, 3) or (predicted_noise, attention_weights) if return_attention=True
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)

        # 1. Prepare embeddings
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        z_latent = self.z_projection(context).view(batch_size, 1, -1)  # (B, 1, z_dim)
        ctx_emb = torch.cat([time_emb, z_latent], dim=-1)  # (B, 1, 3+z_dim)

        # 2. Parse text embeddings
        text_tokens, text_pool, token_ids = None, None, None
        if text_emb is not None:
            if isinstance(text_emb, dict):
                text_tokens = text_emb.get('tokens')    # (B, seq_len, 512) for cross-attention
                text_pool = text_emb.get('pool')        # (B, 512) for FiLM
                token_ids = text_emb.get('token_ids')   # (B, seq_len) for stop word filtering
            else:
                # Backward compatibility
                text_pool = text_emb if text_emb.dim() == 2 else None
                text_tokens = text_emb if text_emb.dim() == 3 else None

        # 3. Forward through MLP with multi-scale cross-attention
        attention_weights_256 = None
        attention_weights_512 = None

        # Layer 0: 3 -> 128
        out = self.layers[0](ctx=ctx_emb, x=x)
        out = self.act(out)

        # Layer 1: 128 -> 256 + Cross-Attention (mid-level: part structures)
        out = self.layers[1](ctx=ctx_emb, x=out)
        if self.use_cross_attention and text_tokens is not None:
            out_norm = self.cross_attn_norm_256(out)
            if return_attention:
                attn_out, attention_weights_256 = self.cross_attn_256(
                    out_norm, text_tokens, return_attention=True,
                    token_ids=token_ids, tokenizer=self.tokenizer
                )
            else:
                attn_out = self.cross_attn_256(
                    out_norm, text_tokens, return_attention=False,
                    token_ids=token_ids, tokenizer=self.tokenizer
                )
            out = out + 0.5 * attn_out
        out = self.act(out)

        # Layer 2: 256 -> 512 + Cross-Attention (high-level)
        out = self.layers[2](ctx=ctx_emb, x=out)
        if self.use_cross_attention and text_tokens is not None:
            out_norm = self.cross_attn_norm_512(out)
            if return_attention:
                attn_out, attention_weights_512 = self.cross_attn_512(
                    out_norm, text_tokens, return_attention=True,
                    token_ids=token_ids, tokenizer=self.tokenizer
                )
            else:
                attn_out = self.cross_attn_512(
                    out_norm, text_tokens, return_attention=False,
                    token_ids=token_ids, tokenizer=self.tokenizer
                )
            out = out + 0.5 * attn_out
        if self.use_film and text_pool is not None:
            out = self.film_1(out, text_pool)
        out = self.act(out)

        # Layer 3: 512 -> 256
        if self.use_film and text_pool is not None:
            out = self.film_2(out, text_pool)
        out = self.layers[3](ctx=ctx_emb, x=out)
        out = self.act(out)

        # Layer 4: 256 -> 128
        out = self.layers[4](ctx=ctx_emb, x=out)
        out = self.act(out)

        # Layer 5: 128 -> 3
        out = self.layers[5](ctx=ctx_emb, x=out)

        # Residual connection and return
        result = x + out if self.residual else out

        if return_attention:
            # Return both 256-dim and 512-dim attention weights for multi-scale analysis
            return result, {'attn_256': attention_weights_256, 'attn_512': attention_weights_512}

        return result


class DiffusionPoint(Module):

    def __init__(self, net, var_sched:VarianceSchedule, use_alignment_loss=True, alignment_weight=0.1):
        super().__init__()
        self.net = net
        self.var_sched = var_sched
        self.use_alignment_loss = use_alignment_loss
        self.alignment_weight = alignment_weight

        # Shape encoder for alignment loss (maps point cloud to CLIP space)
        if use_alignment_loss:
            from .encoders import PointNetEncoder
            self.shape_encoder = PointNetEncoder(zdim=512)  # Match CLIP dimension

    def get_loss(self, x_0, context, text_emb=None, t=None, writer=None, it=None):
        """
        Args:
            x_0:  Input point cloud, (B, N, d).
            context:  Shape latent, (B, F).
            text_emb: Text embeddings from CLIP (dict or tensor). Optional.
        """
        batch_size, _, point_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

        e_rand = torch.randn_like(x_0)  # (B, N, d)
        x_t = c0 * x_0 + c1 * e_rand    # (B, N, d) - noisy point cloud
        e_theta = self.net(x_t, beta=beta, context=context, text_emb=text_emb)

        # Noise prediction loss (L_noise)
        loss_noise = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')

        # Text-shape alignment loss (L_align)
        loss_align = torch.tensor(0.0).to(x_0.device)
        if self.use_alignment_loss and text_emb is not None:
            # Extract text_pool
            text_pool = None
            if isinstance(text_emb, dict):
                text_pool = text_emb.get('pool', None)

            if text_pool is not None:
                # Predict x_0 from x_t and e_theta
                x_0_pred = (x_t - c1 * e_theta) / c0

                # Encode predicted shape to CLIP space
                # Use detach() to prevent gradient backprop through x_0_pred (saves memory)
                z_mu, z_sigma = self.shape_encoder(x_0_pred.detach())
                shape_features = z_mu  # (B, 512)

                # Compute cosine similarity (maximize = negative loss)
                # Normalize features
                shape_features_norm = F.normalize(shape_features, dim=-1)
                text_pool_norm = F.normalize(text_pool, dim=-1)

                # Cosine similarity
                similarity = (shape_features_norm * text_pool_norm).sum(dim=-1)  # (B,)
                loss_align = -similarity.mean()  # Negative: we want to maximize similarity

        # Total loss: L_total = L_noise + λ_align * L_align
        loss_total = loss_noise + self.alignment_weight * loss_align

        # Logging
        if writer is not None and it is not None:
            writer.add_scalar('train/loss_noise', loss_noise.item(), it)
            writer.add_scalar('train/loss_align', loss_align.item(), it)
            writer.add_scalar('train/loss_total', loss_total.item(), it)
            if text_pool is not None:
                writer.add_scalar('train/text_shape_similarity', -loss_align.item(), it)

        return loss_total

    def sample(self, num_points, context, text_emb=None, point_dim=3, flexibility=0.0, ret_traj=False, return_attention=False):
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
        traj = {self.var_sched.num_steps: x_T}
        attention_weights = None

        for t in range(self.var_sched.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t]*batch_size]

            # Get attention weights from the final denoising step
            if return_attention and t == 1:
                e_theta, attention_weights = self.net(x_t, beta=beta, context=context, text_emb=text_emb, return_attention=True)
            else:
                e_theta = self.net(x_t, beta=beta, context=context, text_emb=text_emb)

            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]

        result = traj if ret_traj else traj[0]

        if return_attention:
            return result, attention_weights
        return result
