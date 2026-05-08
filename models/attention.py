import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Linear
import math


# Stop words that should receive lower attention
STOP_WORDS = {
    'a', 'an', 'the', 'with', 'of', 'in', 'on', 'at', 'to', 'for',
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'and', 'or', 'but', 'if', 'while', 'as',
    'this', 'that', 'these', 'those',
    'has', 'have', 'had', 'do', 'does', 'did',
    'it', 'its', 'from', 'by', 'about', 'into', 'through',
    'during', 'before', 'after', 'above', 'below',
}


class CrossAttention(Module):
    """
    Cross-attention module for text-to-point conditioning.
    Query: point features, Key/Value: text token embeddings
    """

    def __init__(self, query_dim, context_dim, heads=8, dim_head=64, dropout=0.0,
                 use_stopword_filter=True, stopword_penalty=0.1):
        """
        Args:
            query_dim: Dimension of query (point features)
            context_dim: Dimension of context (text embeddings)
            heads: Number of attention heads
            dim_head: Dimension per head
            dropout: Dropout probability
            use_stopword_filter: If True, reduce attention to stop words
            stopword_penalty: Multiplier for stop word attention (0.1 = reduce to 10%)
        """
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.use_stopword_filter = use_stopword_filter
        self.stopword_penalty = stopword_penalty

        self.to_q = Linear(query_dim, inner_dim, bias=False)
        self.to_k = Linear(context_dim, inner_dim, bias=False)
        self.to_v = Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context, mask=None, return_attention=False, token_ids=None, tokenizer=None):
        """
        Args:
            x: Query tensor (B, N_points, query_dim)
            context: Context tensor (B, N_tokens, context_dim)
            mask: Optional attention mask (B, N_points, N_tokens)
            return_attention: If True, return attention weights along with output
            token_ids: Token IDs for stop word filtering (B, N_tokens) [optional]
            tokenizer: CLIP tokenizer for decoding tokens [optional]

        Returns:
            If return_attention=False:
                Output tensor (B, N_points, query_dim)
            If return_attention=True:
                (output, attention_weights) where attention_weights is (B, heads, N_points, N_tokens)
        """
        h = self.heads

        # Project to Q, K, V
        q = self.to_q(x)  # (B, N_points, inner_dim)
        k = self.to_k(context)  # (B, N_tokens, inner_dim)
        v = self.to_v(context)  # (B, N_tokens, inner_dim)

        # Reshape for multi-head attention
        # (B, N, inner_dim) -> (B, N, heads, dim_head) -> (B, heads, N, dim_head)
        q = q.reshape(q.size(0), q.size(1), h, -1).permute(0, 2, 1, 3)
        k = k.reshape(k.size(0), k.size(1), h, -1).permute(0, 2, 1, 3)
        v = v.reshape(v.size(0), v.size(1), h, -1).permute(0, 2, 1, 3)

        # Attention: (B, heads, N_points, dim_head) @ (B, heads, dim_head, N_tokens)
        #         -> (B, heads, N_points, N_tokens)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # Mask: (B, N_points, N_tokens) -> (B, 1, N_points, N_tokens)
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)

        # Apply stop word filtering (always active, both training and inference)
        if self.use_stopword_filter and token_ids is not None and tokenizer is not None:
            attn = self.stopword_filter(attn, token_ids, tokenizer)

        # Apply attention to values: (B, heads, N_points, N_tokens) @ (B, heads, N_tokens, dim_head)
        #                          -> (B, heads, N_points, dim_head)
        out = torch.matmul(attn, v)

        # Reshape back: (B, heads, N_points, dim_head) -> (B, N_points, heads*dim_head)
        out = out.permute(0, 2, 1, 3).reshape(out.size(0), out.size(2), -1)

        # Project to output
        output = self.to_out(out)

        if return_attention:
            return output, attn
        return output

    def stopword_filter(self, attn, token_ids, tokenizer):
        """
        Filter out stop words by reducing their attention weights

        Args:
            attn: (B, heads, N_points, N_tokens) attention weights
            token_ids: (B, N_tokens) token IDs
            tokenizer: CLIP tokenizer

        Returns:
            attn_filtered: (B, heads, N_points, N_tokens) filtered attention
        """
        batch_size, n_heads, n_points, n_tokens = attn.shape

        # Create a mask for stop words (1.0 for normal words, stopword_penalty for stop words)
        # Initialize with all 1.0 (no filtering)
        filter_mask = torch.ones(batch_size, n_tokens, device=attn.device, dtype=attn.dtype)

        # Decode tokens and identify stop words
        for b in range(batch_size):
            tids = token_ids[b].cpu().numpy()

            for i, tid in enumerate(tids):
                # Decode token
                token_str = tokenizer.decode([int(tid)]).strip().lower()
                # Remove CLIP artifacts
                token_str = token_str.replace('</w>', '').replace('<|startoftext|>', '').replace('<|endoftext|>', '')

                # If stop word, set mask to penalty value
                if token_str in STOP_WORDS:
                    filter_mask[b, i] = self.stopword_penalty

        # Apply mask: (B, heads, N_points, N_tokens) * (B, 1, 1, N_tokens)
        filter_mask = filter_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N_tokens)
        attn_filtered = attn * filter_mask

        # Re-normalize attention weights
        attn_filtered = attn_filtered / (attn_filtered.sum(dim=-1, keepdim=True) + 1e-8)

        return attn_filtered
