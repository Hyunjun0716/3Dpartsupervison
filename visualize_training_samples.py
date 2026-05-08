"""
Visualize generated samples from training captions and compare with GT
This script:
1. Loads training dataset with captions
2. Generates point clouds from training captions
3. Compares generated samples with GT using CD/EMD metrics
4. Saves visualizations as images
"""

import os
import argparse
import torch
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.dataset import ShapeNetCoreText
from utils.misc import seed_all
from models.vae_gaussian import GaussianVAE
from models.vae_flow import FlowVAE
from models.clip_encoder import FrozenCLIPTextEmbedder
from evaluation.evaluation_metrics import compute_all_metrics, jsd_between_point_cloud_sets, EMD_CD


def visualize_point_cloud(ax, points, title, color='blue', elev=30, azim=45):
    """
    Visualize a single point cloud on given axis

    Args:
        ax: matplotlib 3D axis
        points: (N, 3) point cloud
        title: title for the plot
        color: color of points
        elev: elevation angle
        azim: azimuth angle
    """
    points = points.cpu().numpy() if torch.is_tensor(points) else points

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=color, s=1, alpha=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set equal aspect ratio
    max_range = np.abs(points).max()
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    ax.view_init(elev=elev, azim=azim)


def normalize_point_cloud(pc, mode='shape_unit'):
    """
    Normalize point cloud

    Args:
        pc: (N, 3) point cloud tensor
        mode: normalization mode
    """
    if mode == 'shape_unit':
        shift = pc.mean(dim=0, keepdim=True)
        scale = pc.flatten().std()
    elif mode == 'shape_bbox':
        pc_max = pc.max(dim=0, keepdim=True)[0]
        pc_min = pc.min(dim=0, keepdim=True)[0]
        shift = (pc_min + pc_max) / 2
        scale = (pc_max - pc_min).max() / 2
    else:
        return pc

    pc = (pc - shift) / scale
    return pc


def visualize_attention_weights(attention_weights, tokens, caption, save_path, tokenizer, top_k=10):
    """
    Visualize attention weights as a heatmap showing which words the model focuses on.

    Args:
        attention_weights: (B, heads, N_points, N_tokens) or (N_points, N_tokens) attention tensor
        tokens: Token IDs tensor from CLIP tokenizer
        caption: Original caption text
        save_path: Path to save the visualization
        tokenizer: CLIP tokenizer instance for proper decoding
        top_k: Number of top tokens to highlight
    """
    # Convert to numpy and average over heads and points if needed
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.cpu().numpy()

    # Handle different shapes
    if len(attention_weights.shape) == 4:
        # (B, heads, N_points, N_tokens) -> average over batch, heads, and points
        attn = attention_weights[0].mean(axis=0).mean(axis=0)  
    elif len(attention_weights.shape) == 3:
        # (heads, N_points, N_tokens) -> average over heads and points
        attn = attention_weights.mean(axis=0).mean(axis=0)  # (N_tokens,)
    elif len(attention_weights.shape) == 2:
        # (N_points, N_tokens) -> average over points
        attn = attention_weights.mean(axis=0)  # (N_tokens,)
    else:
        attn = attention_weights

    # Get actual token IDs and decode them
    if isinstance(tokens, torch.Tensor):
        token_ids = tokens.cpu().numpy()
    else:
        token_ids = np.array(tokens)

    # Handle batch dimension
    if len(token_ids.shape) == 2:
        token_ids = token_ids[0]  # Take first batch

    # Decode each token to get actual token strings
    token_strings = []
    for token_id in token_ids:
        token_str = tokenizer.decode([int(token_id)]).strip()
        token_strings.append(token_str)

    # Find where actual content ends (before padding)
    # CLIP uses token_id 49407 for end-of-text [SEP] and 0 or 49407 for padding
    end_token_id = 49407  # CLIP's [SEP] token
    try:
        # Find first [SEP] or padding token
        if end_token_id in token_ids:
            end_idx = np.where(token_ids == end_token_id)[0][0]
        else:
            end_idx = len(token_ids)
    except:
        end_idx = len(token_ids)

    # Extract content tokens (skip [CLS] at position 0, and [SEP]/padding after end_idx)
    start_idx = 1
    content_token_ids = token_ids[start_idx:end_idx]
    content_tokens = token_strings[start_idx:end_idx]
    content_attention = attn[start_idx:end_idx]

    # Normalize attention weights (only for content tokens)
    content_attention = content_attention / (content_attention.sum() + 1e-8)

    # Clean up token strings (remove special characters from CLIP tokenization)
    cleaned_tokens = []
    for token in content_tokens:
        # CLIP tokenizer adds '</w>' for word boundaries
        token_clean = token.replace('</w>', '').strip()
        if token_clean:
            cleaned_tokens.append(token_clean)
        else:
            cleaned_tokens.append(token)

    word_attention = content_attention[:len(cleaned_tokens)]
    word_labels = cleaned_tokens

    # Ensure we have valid data
    if len(word_attention) == 0 or len(word_labels) == 0:
        print(f"[WARNING] No valid attention data for visualization")
        return

    # Sort by attention weight and get top_k
    top_indices = np.argsort(word_attention)[-top_k:][::-1]
    top_words = [word_labels[i] for i in top_indices]
    top_weights = [word_attention[i] for i in top_indices]

    # Create visualization
    # Dynamically adjust figure height based on number of words to prevent overlap
    num_words = len(word_labels)
    fig_height = max(8, num_words * 0.4)  # At least 8 inches, or 0.4 inch per word
    fig, ax1= plt.subplots(1, 1, figsize=(12, fig_height))

    # Adjust font size based on number of words
    if num_words <= 15:
        fontsize = 10
    elif num_words <= 30:
        fontsize = 8
    else:
        fontsize = 6

    # Truncate long words to prevent horizontal overflow
    max_word_len = 20
    display_labels = [w if len(w) <= max_word_len else w[:max_word_len-3] + '...'
                      for w in word_labels]

    # Plot 1: Bar chart of all words
    colors = ['red' if i in top_indices else 'lightblue' for i in range(len(word_labels))]
    bars = ax1.barh(range(len(word_labels)), word_attention, color=colors)
    ax1.set_yticks(range(len(word_labels)))
    ax1.set_yticklabels(display_labels, fontsize=fontsize)
    ax1.set_xlabel('Attention Weight', fontsize=12)
    display_caption = caption if len(caption) <= 60 else caption[:60] + "..."
    ax1.set_title(f'Word Attention Weights\nCaption: "{display_caption}"',
            fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    # Add value labels on bars
    for i, (bar, weight) in enumerate(zip(bars, word_attention)):
        if i in top_indices:
            ax1.text(weight, i, f' {weight:.3f}', va='center', fontweight='bold')

    # Add text summary
    top_words_str = ", ".join([f"{w} ({top_weights[i]:.3f})" for i, w in enumerate(top_words)])
    fig.text(0.5, 0.02, f'Top {top_k} Important Words: {top_words_str}',
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[INFO] Saved attention visualization: {save_path}")


def visualize_token_point_heatmap(attention_weights, tokens, caption, save_path, tokenizer, top_k_words=5, top_k_points=100):
    """
    Level 2: Token→Point Heatmap
    Show which points attend to which words

    Args:
        attention_weights: (B, heads, N_points, N_tokens) or (heads, N_points, N_tokens)
        tokens: Token IDs tensor
        caption: Original caption
        save_path: Path to save visualization
        tokenizer: CLIP tokenizer
        top_k_words: Number of top words to show
        top_k_points: Number of points to show in heatmap
    """
    import seaborn as sns

    # Check if attention weights is None
    if attention_weights is None:
        print(f"[WARNING] No valid attention data for visualization")
        return

    # Process attention weights
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.cpu().numpy()

    # Handle shape: (B, heads, N_points, N_tokens) -> (N_points, N_tokens)
    if len(attention_weights.shape) == 4:
        attn = attention_weights[0].mean(axis=0)  # Average over batch and heads
    elif len(attention_weights.shape) == 3:
        attn = attention_weights.mean(axis=0)  # Average over heads
    else:
        attn = attention_weights

    # Check if attention is empty or has invalid shape
    if attn.size == 0 or attn.shape[0] == 0 or attn.shape[1] == 0:
        print(f"[WARNING] Empty attention data for visualization")
        return

    # Get token strings
    if isinstance(tokens, torch.Tensor):
        token_ids = tokens.cpu().numpy()
    else:
        token_ids = np.array(tokens)

    if len(token_ids.shape) == 2:
        token_ids = token_ids[0]

    # Decode tokens
    token_strings = []
    for token_id in token_ids:
        token_str = tokenizer.decode([int(token_id)]).strip()
        token_strings.append(token_str)

    # Find content tokens (skip [CLS] and [SEP]/padding)
    end_token_id = 49407
    try:
        end_idx = np.where(token_ids == end_token_id)[0][0] if end_token_id in token_ids else len(token_ids)
    except:
        end_idx = len(token_ids)

    start_idx = 1
    content_tokens = [t.replace('</w>', '').strip() for t in token_strings[start_idx:end_idx]]
    content_attention = attn[:, start_idx:end_idx]  # (N_points, N_content_tokens)

    # Check if we have valid content tokens
    if len(content_tokens) == 0 or content_attention.shape[1] == 0:
        print(f"[WARNING] No valid content tokens for visualization")
        return

    # Select top-K words by average attention
    word_importance = content_attention.mean(axis=0)  # (N_content_tokens,)
    actual_top_k_words = min(top_k_words, len(content_tokens))

    if actual_top_k_words == 0:
        print(f"[WARNING] No words available for visualization")
        return

    top_word_indices = np.argsort(word_importance)[-actual_top_k_words:][::-1]

    selected_words = [content_tokens[i] for i in top_word_indices]
    selected_attention = content_attention[:, top_word_indices]  # (N_points, top_k_words)

    # Select top-K points (points with highest total attention to selected words)
    point_importance = selected_attention.sum(axis=1)  # (N_points,)
    actual_top_k_points = min(top_k_points, selected_attention.shape[0])

    if actual_top_k_points == 0:
        print(f"[WARNING] No points available for visualization")
        return

    top_point_indices = np.argsort(point_importance)[-actual_top_k_points:][::-1]

    heatmap_data = selected_attention[top_point_indices, :]  # (top_k_points, top_k_words)

    # Check if heatmap data is valid
    if heatmap_data.size == 0:
        print(f"[WARNING] Empty heatmap data for visualization")
        return

    # Normalize each row (per point)
    row_sums = heatmap_data.sum(axis=1, keepdims=True)
    heatmap_data = heatmap_data / (row_sums + 1e-8)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(8, actual_top_k_words * 1.5), max(6, actual_top_k_points * 0.1)))
    sns.heatmap(heatmap_data,
                xticklabels=selected_words,
                yticklabels=[f'P{i}' for i in top_point_indices],
                cmap='YlOrRd',
                cbar_kws={'label': 'Attention Weight'},
                ax=ax)

    ax.set_xlabel('Words', fontsize=12)
    ax.set_ylabel('Point Index', fontsize=12)
    display_caption = caption if len(caption) <= 60 else caption[:60] + "..."
    ax.set_title(f'Token→Point Attention Heatmap (Top {actual_top_k_words} words, Top {actual_top_k_points} points)\nCaption: "{display_caption}"',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[INFO] Saved token-point heatmap: {save_path}")


def visualize_token_point_3d(attention_weights, tokens, caption, point_cloud, save_path, tokenizer, top_k_words=5):
    """
    Level 2 Enhanced: 3D Point Cloud Visualization with Word-based Attention
    Show which spatial regions of the point cloud attend to which words

    Args:
        attention_weights: (B, heads, N_points, N_tokens) or (heads, N_points, N_tokens)
        tokens: Token IDs tensor
        caption: Original caption
        point_cloud: (N_points, 3) point cloud coordinates
        save_path: Path to save visualization
        tokenizer: CLIP tokenizer
        top_k_words: Number of top words to visualize
    """
    # Check if attention weights is None
    if attention_weights is None:
        print(f"[WARNING] No valid attention data for 3D visualization")
        return

    # Process attention weights
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.cpu().numpy()

    # Handle shape: (B, heads, N_points, N_tokens) -> (N_points, N_tokens)
    if len(attention_weights.shape) == 4:
        attn = attention_weights[0].mean(axis=0)  # Average over batch and heads
    elif len(attention_weights.shape) == 3:
        attn = attention_weights.mean(axis=0)  # Average over heads
    else:
        attn = attention_weights

    # Check if attention is empty or has invalid shape
    if attn.size == 0 or attn.shape[0] == 0 or attn.shape[1] == 0:
        print(f"[WARNING] Empty attention data for 3D visualization")
        return

    # Process point cloud
    if torch.is_tensor(point_cloud):
        point_cloud = point_cloud.cpu().numpy()

    # Get token strings
    if isinstance(tokens, torch.Tensor):
        token_ids = tokens.cpu().numpy()
    else:
        token_ids = np.array(tokens)

    if len(token_ids.shape) == 2:
        token_ids = token_ids[0]

    # Decode tokens
    token_strings = []
    for token_id in token_ids:
        token_str = tokenizer.decode([int(token_id)]).strip()
        token_strings.append(token_str)

    # Find content tokens (skip [CLS] and [SEP]/padding)
    end_token_id = 49407
    try:
        end_idx = np.where(token_ids == end_token_id)[0][0] if end_token_id in token_ids else len(token_ids)
    except:
        end_idx = len(token_ids)

    start_idx = 1
    content_tokens = [t.replace('</w>', '').strip() for t in token_strings[start_idx:end_idx]]
    content_attention = attn[:, start_idx:end_idx]  # (N_points, N_content_tokens)

    # Check if we have valid content tokens
    if len(content_tokens) == 0 or content_attention.shape[1] == 0:
        print(f"[WARNING] No valid content tokens for 3D visualization")
        return

    # Select top-K words by average attention
    word_importance = content_attention.mean(axis=0)  # (N_content_tokens,)
    actual_top_k_words = min(top_k_words, len(content_tokens))

    if actual_top_k_words == 0:
        print(f"[WARNING] No words available for 3D visualization")
        return

    top_word_indices = np.argsort(word_importance)[-actual_top_k_words:][::-1]

    selected_words = [content_tokens[i] for i in top_word_indices]

    # Create figure with subplots for each word
    n_rows = (actual_top_k_words + 1) // 2  # 2 columns
    n_cols = 2
    fig = plt.figure(figsize=(14, 6 * n_rows))

    for idx, (word_idx, word) in enumerate(zip(top_word_indices, selected_words)):
        # Get attention for this word
        word_attn = content_attention[:, word_idx]  # (N_points,)

        # Normalize to [0, 1]
        word_attn_norm = (word_attn - word_attn.min()) / (word_attn.max() - word_attn.min() + 1e-8)

        # Create 3D subplot
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')

        # Plot point cloud with attention-based colors
        scatter = ax.scatter(
            point_cloud[:, 0],
            point_cloud[:, 1],
            point_cloud[:, 2],
            c=word_attn_norm,
            cmap='RdYlBu_r',  # Red=high attention, Blue=low attention
            s=10,
            alpha=0.7,
            vmin=0,
            vmax=1
        )

        # Set title
        ax.set_title(f'Attention to "{word}"\n(Avg: {word_attn.mean():.3f}, Max: {word_attn.max():.3f})',
                     fontsize=11, fontweight='bold')

        # Set labels
        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Y', fontsize=9)
        ax.set_zlabel('Z', fontsize=9)

        # Set equal aspect ratio
        max_range = np.abs(point_cloud).max()
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

        # Set viewing angle
        ax.view_init(elev=25, azim=45)

        # Add colorbar for first subplot only
        if idx == 0:
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=10)
            cbar.set_label('Attention Weight', fontsize=9)

    # Add overall caption
    display_caption = caption if len(caption) <= 100 else caption[:100] + "..."
    fig.suptitle(f'3D Point Cloud Attention Visualization (Top {actual_top_k_words} words)\nCaption: "{display_caption}"',
                 fontsize=13, fontweight='bold', y=0.98)

    # Add color interpretation note
    fig.text(0.5, 0.02, 'Color: Red = High Attention (model focuses on these points for this word), Blue = Low Attention',
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[INFO] Saved 3D token-point visualization: {save_path}")


def visualize_multiscale_comparison(attn_256, attn_512, tokens, caption, save_path, tokenizer, top_k=10):
    """
    Level 3: Multi-scale Attention Comparison
    Compare 256-dim (local/part) vs 512-dim (global) attention

    Args:
        attn_256: 256-dim attention weights (B, heads, N_points, N_tokens)
        attn_512: 512-dim attention weights (B, heads, N_points, N_tokens)
        tokens: Token IDs
        caption: Original caption
        save_path: Path to save visualization
        tokenizer: CLIP tokenizer
        top_k: Number of top words to highlight
    """
    from scipy.stats import entropy

    # Check if attention weights are None
    if attn_256 is None or attn_512 is None:
        print(f"[WARNING] No valid attention data for multi-scale comparison")
        return

    # Process both attention weights
    def process_attn(attn):
        if torch.is_tensor(attn):
            attn = attn.cpu().numpy()
        if len(attn.shape) == 4:
            return attn[0].mean(axis=0).mean(axis=0)  # (N_tokens,)
        elif len(attn.shape) == 3:
            return attn.mean(axis=0).mean(axis=0)
        return attn.mean(axis=0) if len(attn.shape) == 2 else attn

    attn_256_vec = process_attn(attn_256)
    attn_512_vec = process_attn(attn_512)

    # Check if processed attention is empty
    if attn_256_vec.size == 0 or attn_512_vec.size == 0:
        print(f"[WARNING] Empty attention data for multi-scale comparison")
        return

    # Get token strings
    if isinstance(tokens, torch.Tensor):
        token_ids = tokens.cpu().numpy()
    else:
        token_ids = np.array(tokens)

    if len(token_ids.shape) == 2:
        token_ids = token_ids[0]

    token_strings = [tokenizer.decode([int(tid)]).strip() for tid in token_ids]

    # Extract content tokens
    end_token_id = 49407
    try:
        end_idx = np.where(token_ids == end_token_id)[0][0] if end_token_id in token_ids else len(token_ids)
    except:
        end_idx = len(token_ids)

    start_idx = 1
    content_tokens = [t.replace('</w>', '').strip() for t in token_strings[start_idx:end_idx]]
    attn_256_content = attn_256_vec[start_idx:end_idx]
    attn_512_content = attn_512_vec[start_idx:end_idx]

    # Check if we have valid content tokens
    if len(content_tokens) == 0 or attn_256_content.size == 0 or attn_512_content.size == 0:
        print(f"[WARNING] No valid content tokens for multi-scale comparison")
        return

    # Normalize
    attn_256_content = attn_256_content / (attn_256_content.sum() + 1e-8)
    attn_512_content = attn_512_content / (attn_512_content.sum() + 1e-8)

    # Compute KL divergence
    kl_256_to_512 = entropy(attn_256_content + 1e-8, attn_512_content + 1e-8)
    kl_512_to_256 = entropy(attn_512_content + 1e-8, attn_256_content + 1e-8)

    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Side-by-side bar chart
    ax1 = axes[0, 0]
    x = np.arange(len(content_tokens))
    width = 0.35
    ax1.bar(x - width/2, attn_256_content, width, label='256-dim (Local/Part)', color='skyblue', alpha=0.8)
    ax1.bar(x + width/2, attn_512_content, width, label='512-dim (Global)', color='salmon', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(content_tokens, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Attention Weight')
    ax1.set_title('256-dim vs 512-dim Attention Comparison')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Difference (512 - 256)
    ax2 = axes[0, 1]
    diff = attn_512_content - attn_256_content
    colors = ['green' if d > 0 else 'red' for d in diff]
    ax2.bar(x, diff, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(content_tokens, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Attention Difference')
    ax2.set_title('Attention Difference (512-dim minus 256-dim)\nGreen: More global, Red: More local')
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Top-K words for 256-dim
    ax3 = axes[1, 0]
    top_indices_256 = np.argsort(attn_256_content)[-top_k:][::-1]
    top_words_256 = [content_tokens[i] for i in top_indices_256]
    top_weights_256 = attn_256_content[top_indices_256]
    ax3.barh(range(len(top_words_256)), top_weights_256, color='skyblue', alpha=0.8)
    ax3.set_yticks(range(len(top_words_256)))
    ax3.set_yticklabels(top_words_256)
    ax3.set_xlabel('Attention Weight')
    ax3.set_title(f'Top-{top_k} Words: 256-dim (Local/Part Focus)')
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3)

    # Plot 4: Top-K words for 512-dim
    ax4 = axes[1, 1]
    top_indices_512 = np.argsort(attn_512_content)[-top_k:][::-1]
    top_words_512 = [content_tokens[i] for i in top_indices_512]
    top_weights_512 = attn_512_content[top_indices_512]
    ax4.barh(range(len(top_words_512)), top_weights_512, color='salmon', alpha=0.8)
    ax4.set_yticks(range(len(top_words_512)))
    ax4.set_yticklabels(top_words_512)
    ax4.set_xlabel('Attention Weight')
    ax4.set_title(f'Top-{top_k} Words: 512-dim (Global Focus)')
    ax4.invert_yaxis()
    ax4.grid(axis='x', alpha=0.3)

    # Add overall statistics
    display_caption = caption if len(caption) <= 80 else caption[:80] + "..."
    stats_text = f'Caption: "{display_caption}"\n'
    stats_text += f'KL Divergence (256→512): {kl_256_to_512:.4f} | KL Divergence (512→256): {kl_512_to_256:.4f}\n'
    stats_text += f'Higher KL = More different attention patterns'

    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[INFO] Saved multi-scale comparison: {save_path}")
    print(f"       KL(256→512): {kl_256_to_512:.4f}, KL(512→256): {kl_512_to_256:.4f}")


def main(args):
    # Set seed
    seed_all(args.seed)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[INFO] Loading checkpoint from: {args.ckpt}")

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=args.device)

    print(f"[INFO] Loaded checkpoint from iteration {ckpt.get('iteration', 'unknown')}")

    # Load CLIP text encoder BEFORE model (for stop word filtering)
    use_text = ckpt['args'].use_text_condition if hasattr(ckpt['args'], 'use_text_condition') else False
    if use_text:
        print("[INFO] Loading CLIP text encoder...")
        clip_model = ckpt['args'].clip_model if hasattr(ckpt['args'], 'clip_model') else 'openai/clip-vit-base-patch32'
        text_encoder = FrozenCLIPTextEmbedder(version=clip_model, device=args.device, return_sequence=True)
        text_encoder = text_encoder.to(args.device)  # Move to GPU
        text_encoder.eval()
    else:
        text_encoder = None
        print("[WARNING] Model was trained without text conditioning!")

    # Determine model type from checkpoint and pass tokenizer
    if 'latent_flow_depth' in ckpt['args']:
        print("[INFO] Model type: FlowVAE")
        model = FlowVAE(ckpt['args'], tokenizer=text_encoder.tokenizer if text_encoder else None).to(args.device)
    else:
        print("[INFO] Model type: GaussianVAE")
        model = GaussianVAE(ckpt['args'], tokenizer=text_encoder.tokenizer if text_encoder else None).to(args.device)

    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # Use custom captions or load dataset
    if args.interactive:
        # Interactive mode: prompt user for captions
        print("\n" + "="*80)
        print("INTERACTIVE CAPTION MODE")
        print("="*80)
        print("Enter captions one by one. Press Enter without text to finish.")
        print("Examples:")
        print("  - a wooden chair with four legs")
        print("  - a modern office chair with wheels and armrests")
        print("  - a minimalist dining chair with curved back")
        print("-"*80)

        all_captions = []
        while True:
            caption = input(f"\nCaption {len(all_captions) + 1} (or press Enter to finish): ").strip()
            if not caption:
                break
            all_captions.append(caption)
            print(f"  ✓ Added: \"{caption}\"")

        if len(all_captions) == 0:
            print("[ERROR] No captions provided. Exiting.")
            return

        print(f"\n[INFO] Total captions: {len(all_captions)}")
        num_samples = len(all_captions)
        use_dataset = False
    elif args.custom_captions is not None:
        print(f"[INFO] Using {len(args.custom_captions)} custom captions")
        all_captions = args.custom_captions
        num_samples = len(all_captions)
        use_dataset = False
    else:
        # Load training dataset
        print(f"[INFO] Loading dataset from: {args.dataset_path}")
        train_dataset = ShapeNetCoreText(
            path=args.dataset_path,
            cates=args.categories,
            split='test',
            scale_mode=args.normalize,
            captions_path=args.captions_path,
        )

        print(f"[INFO] Dataset size: {len(train_dataset)}")
        # ShapeNetCoreText already filters to only caption-matched samples
        print(f"[INFO] All samples have matching captions (pre-filtered by dataset)")

        # Select samples to visualize
        num_samples = min(args.num_samples, len(train_dataset))

        # Sample indices (evenly spaced or random)
        if args.sample_mode == 'random':
            indices = np.random.choice(len(train_dataset), num_samples, replace=False)
        else:  # 'evenly'
            indices = np.linspace(0, len(train_dataset) - 1, num_samples, dtype=int)

        use_dataset = True

    print(f"[INFO] Generating {num_samples} samples...")

    # Store results
    all_gt = []
    all_generated = []
    caption_list = []
    all_cd_distances = []
    all_emd_distances = []
    all_attention_weights = []  # Store attention weights for visualization
    all_token_ids = []  # Store token IDs for proper visualization

    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Generating samples"):
            if use_dataset:
                # Get GT data from dataset
                data = train_dataset[indices[i]]
                gt_pc = data['pointcloud'].to(args.device)  # (N, 3)
                caption = data.get('caption', 'No caption')
                model_id = data.get('model_id', 'unknown')
            else:
                # Using custom captions - no GT available
                gt_pc = None
                caption = all_captions[i]

            # Encode text and get token IDs
            token_ids_for_viz = None
            if text_encoder is not None and caption != 'No caption':
                # Get token IDs for visualization
                tokens = text_encoder.tokenizer(
                    [caption],
                    truncation=True,
                    max_length=text_encoder.max_length,
                    return_length=True,
                    return_overflowing_tokens=False,
                    padding="max_length",
                    return_tensors="pt"
                )
                token_ids_for_viz = tokens["input_ids"]

                # Get text embeddings for generation
                text_emb = text_encoder([caption])
            else:
                text_emb = None

            all_token_ids.append(token_ids_for_viz)

            # Sample from model
            # Note: We sample random z from prior, not encoding GT
            # This tests the model's ability to generate from text + random latent
            latent_dim = ckpt['args'].latent_dim
            z = torch.randn(1, latent_dim).to(args.device)

            # Generate with attention weights if text conditioning is used
            if text_emb is not None and args.visualize_attention:
                result = model.sample(
                    z,
                    num_points=args.sample_num_points,
                    flexibility=args.flexibility,
                    text_emb=text_emb,
                    return_attention=True
                )
                if isinstance(result, tuple):
                    generated_pc, attention_weights = result
                    all_attention_weights.append(attention_weights)
                else:
                    generated_pc = result
                    all_attention_weights.append(None)
            else:
                generated_pc = model.sample(
                    z,
                    num_points=args.sample_num_points,
                    flexibility=args.flexibility,
                    text_emb=text_emb
                )  # (1, N, 3)
                all_attention_weights.append(None)

            generated_pc = generated_pc.squeeze(0)  # (N, 3)

            # Compute metrics only if GT is available
            if gt_pc is not None:
                # Ensure same number of points for fair comparison
                if gt_pc.shape[0] != generated_pc.shape[0]:
                    # Resample GT to match generated
                    perm = torch.randperm(gt_pc.shape[0])[:generated_pc.shape[0]]
                    gt_pc_resampled = gt_pc[perm]
                else:
                    gt_pc_resampled = gt_pc

                # Compute CD and EMD
                metrics = EMD_CD(
                    generated_pc.unsqueeze(0),
                    gt_pc_resampled.unsqueeze(0),
                    batch_size=1
                )

                cd_dist = metrics['MMD-CD'].item()
                emd_dist = metrics['MMD-EMD'].item()

                all_gt.append(gt_pc_resampled.cpu())
            else:
                cd_dist = None
                emd_dist = None

            # Store results
            all_generated.append(generated_pc.cpu())
            caption_list.append(caption)
            all_cd_distances.append(cd_dist)
            all_emd_distances.append(emd_dist)

    # Compute statistics (only if GT is available)
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    if use_dataset:
        # Filter out None values
        valid_cd = [cd for cd in all_cd_distances if cd is not None]
        valid_emd = [emd for emd in all_emd_distances if emd is not None]

        if valid_cd:
            print(f"Average Chamfer Distance (CD):  {np.mean(valid_cd):.6f} ± {np.std(valid_cd):.6f}")
            print(f"Average Earth Mover Distance:    {np.mean(valid_emd):.6f} ± {np.std(valid_emd):.6f}")
            print(f"Min CD: {np.min(valid_cd):.6f}")
            print(f"Max CD: {np.max(valid_cd):.6f}")
    else:
        print("Custom captions used - no ground truth metrics available")
        print(f"Generated {num_samples} point clouds from custom captions")
    print("="*80)

    # Save statistics to file
    stats_file = os.path.join(args.save_dir, 'statistics.txt')
    with open(stats_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write(f"Mode: {'Dataset' if use_dataset else 'Custom captions'}\n\n")

        if use_dataset:
            valid_cd = [cd for cd in all_cd_distances if cd is not None]
            valid_emd = [emd for emd in all_emd_distances if emd is not None]
            if valid_cd:
                f.write(f"Average Chamfer Distance (CD):  {np.mean(valid_cd):.6f} ± {np.std(valid_cd):.6f}\n")
                f.write(f"Average Earth Mover Distance:    {np.mean(valid_emd):.6f} ± {np.std(valid_emd):.6f}\n")
                f.write(f"Min CD: {np.min(valid_cd):.6f}\n")
                f.write(f"Max CD: {np.max(valid_cd):.6f}\n")
        else:
            f.write("Custom captions used - no ground truth metrics available\n")

        f.write("="*80 + "\n\n")

        # Per-sample details
        f.write("PER-SAMPLE DETAILS\n")
        f.write("="*80 + "\n")
        for i, (caption, cd, emd) in enumerate(zip(caption_list, all_cd_distances, all_emd_distances)):
            f.write(f"Sample {i}:\n")
            f.write(f"  Caption: {caption}\n")
            if cd is not None:
                f.write(f"  CD:  {cd:.6f}\n")
                f.write(f"  EMD: {emd:.6f}\n")
            else:
                f.write(f"  CD:  N/A (custom caption)\n")
                f.write(f"  EMD: N/A (custom caption)\n")
            f.write("\n")

    print(f"[INFO] Statistics saved to: {stats_file}")

    # Visualize samples
    print(f"\n[INFO] Creating visualizations...")

    samples_per_page = args.samples_per_page
    num_pages = (num_samples + samples_per_page - 1) // samples_per_page

    for page in range(num_pages):
        start_idx = page * samples_per_page
        end_idx = min(start_idx + samples_per_page, num_samples)
        page_samples = end_idx - start_idx

        if use_dataset:
            # Create figure with subplots (2 columns: GT and Generated)
            fig = plt.figure(figsize=(12, 4 * page_samples))

            for i in range(page_samples):
                sample_idx = start_idx + i
                caption = caption_list[sample_idx]
                cd = all_cd_distances[sample_idx]

                # GT point cloud
                ax1 = fig.add_subplot(page_samples, 2, 2*i + 1, projection='3d')
                visualize_point_cloud(
                    ax1,
                    all_gt[sample_idx],
                    f"Ground Truth #{sample_idx}\n\"{caption[:50]}...\"" if len(caption) > 50 else f"Ground Truth #{sample_idx}\n\"{caption}\"",
                    color='blue'
                )

                # Generated point cloud
                ax2 = fig.add_subplot(page_samples, 2, 2*i + 2, projection='3d')
                visualize_point_cloud(
                    ax2,
                    all_generated[sample_idx],
                    f"Generated from Caption #{sample_idx}\nCD: {cd:.4f}",
                    color='red'
                )
        else:
            # For custom captions, only show generated (no GT)
            fig = plt.figure(figsize=(12, 4 * page_samples))

            for i in range(page_samples):
                sample_idx = start_idx + i
                caption = caption_list[sample_idx]

                # Generated point cloud (single column layout)
                ax = fig.add_subplot(page_samples, 1, i + 1, projection='3d')
                visualize_point_cloud(
                    ax,
                    all_generated[sample_idx],
                    f"Generated #{sample_idx}\n\"{caption[:70]}...\"" if len(caption) > 70 else f"Generated #{sample_idx}\n\"{caption}\"",
                    color='red'
                )

        plt.tight_layout()

        # Save figure
        output_file = os.path.join(args.save_dir, f'comparison_page_{page+1}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[INFO] Saved visualization page {page+1}/{num_pages}: {output_file}")

    # Create a summary figure with best and worst samples (only if using dataset)
    if use_dataset:
        print(f"\n[INFO] Creating best/worst samples visualization...")

        # Filter out None values and get valid indices
        valid_indices = [i for i, cd in enumerate(all_cd_distances) if cd is not None]
        valid_cd = [all_cd_distances[i] for i in valid_indices]

        if len(valid_indices) >= 3:
            # Sort by CD
            sorted_idx = np.argsort(valid_cd)
            best_indices = [valid_indices[i] for i in sorted_idx[:3]]  # Top 3 best
            worst_indices = [valid_indices[i] for i in sorted_idx[-3:]]  # Top 3 worst

            fig = plt.figure(figsize=(12, 16))

            for i, idx in enumerate(best_indices):
                caption = caption_list[idx]
                cd = all_cd_distances[idx]

                # GT
                ax1 = fig.add_subplot(6, 2, 2*i + 1, projection='3d')
                visualize_point_cloud(
                    ax1,
                    all_gt[idx],
                    f"BEST #{i+1} - Ground Truth\n\"{caption[:35]}...\"" if len(caption) > 35 else f"BEST #{i+1} - Ground Truth\n\"{caption}\"",
                    color='blue'
                )

                # Generated
                ax2 = fig.add_subplot(6, 2, 2*i + 2, projection='3d')
                visualize_point_cloud(
                    ax2,
                    all_generated[idx],
                    f"BEST #{i+1} - Generated\nCD: {cd:.4f}",
                    color='green'
                )

            for i, idx in enumerate(worst_indices):
                caption = caption_list[idx]
                cd = all_cd_distances[idx]

                # GT
                ax1 = fig.add_subplot(6, 2, 6 + 2*i + 1, projection='3d')
                visualize_point_cloud(
                    ax1,
                    all_gt[idx],
                    f"WORST #{i+1} - Ground Truth\n\"{caption[:35]}...\"" if len(caption) > 35 else f"WORST #{i+1} - Ground Truth\n\"{caption}\"",
                    color='blue'
                )

                # Generated
                ax2 = fig.add_subplot(6, 2, 6 + 2*i + 2, projection='3d')
                visualize_point_cloud(
                    ax2,
                    all_generated[idx],
                    f"WORST #{i+1} - Generated\nCD: {cd:.4f}",
                    color='orange'
                )

            plt.tight_layout()
            summary_file = os.path.join(args.save_dir, 'best_worst_comparison.png')
            plt.savefig(summary_file, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"[INFO] Saved best/worst comparison: {summary_file}")
        else:
            print(f"[WARNING] Not enough samples ({len(valid_indices)}) for best/worst comparison")
    else:
        print(f"\n[INFO] Skipping best/worst comparison (custom captions mode)")

    # Visualize attention weights (3-Level Analysis)
    if args.visualize_attention:
        print(f"\n[INFO] Creating attention visualizations (3-Level Analysis)...")
        attn_dir = os.path.join(args.save_dir, 'attention')
        os.makedirs(attn_dir, exist_ok=True)

        for i, (caption, attention_weights, token_ids) in enumerate(zip(caption_list, all_attention_weights, all_token_ids)):
            if attention_weights is not None and token_ids is not None:
                try:
                    # attention_weights is now a dict: {'attn_256': ..., 'attn_512': ...}
                    # For legacy checkpoints, it might be a single tensor or dict with different keys
                    if isinstance(attention_weights, dict):
                        attn_256 = attention_weights.get('attn_256')
                        attn_512 = attention_weights.get('attn_512')

                        # Legacy checkpoint might have 'attn' key instead
                        attn_legacy = attention_weights.get('attn')

                        # Use legacy attention if multi-scale not available
                        if attn_512 is None and attn_legacy is not None:
                            print(f"[INFO] Using legacy single-scale attention for sample {i}")
                            attn_512 = attn_legacy
                            attn_256 = attn_legacy  # Use same for both

                        # Skip if no valid attention data
                        if attn_512 is None:
                            print(f"[WARNING] No valid attention data for sample {i}, skipping")
                            continue

                        # Level 1: Global Token Weights (using 512-dim for global view)
                        save_path_l1 = os.path.join(attn_dir, f'level1_global_weights_sample_{i:04d}.png')
                        visualize_attention_weights(
                            attn_512,
                            tokens=token_ids,
                            caption=caption,
                            save_path=save_path_l1,
                            tokenizer=text_encoder.tokenizer,
                            top_k=min(10, len(caption.split()))
                        )

                        # Level 2a: Token→Point Heatmap (using 512-dim)
                        save_path_l2a = os.path.join(attn_dir, f'level2a_token_point_heatmap_sample_{i:04d}.png')
                        visualize_token_point_heatmap(
                            attn_512,
                            tokens=token_ids,
                            caption=caption,
                            save_path=save_path_l2a,
                            tokenizer=text_encoder.tokenizer,
                            top_k_words=5,
                            top_k_points=100
                        )

                        # Level 2b: 3D Point Cloud with Attention (using 512-dim)
                        save_path_l2b = os.path.join(attn_dir, f'level2b_3d_point_cloud_sample_{i:04d}.png')
                        visualize_token_point_3d(
                            attn_512,
                            tokens=token_ids,
                            caption=caption,
                            point_cloud=all_generated[i],  # Use generated point cloud
                            save_path=save_path_l2b,
                            tokenizer=text_encoder.tokenizer,
                            top_k_words=5
                        )

                        # Level 3: Multi-scale Comparison (256-dim vs 512-dim)
                        # Skip if using legacy single-scale attention
                        if attn_256 is not None and attn_256 is not attn_512:
                            save_path_l3 = os.path.join(attn_dir, f'level3_multiscale_comparison_sample_{i:04d}.png')
                            visualize_multiscale_comparison(
                                attn_256,
                                attn_512,
                                tokens=token_ids,
                                caption=caption,
                                save_path=save_path_l3,
                                tokenizer=text_encoder.tokenizer,
                                top_k=10
                            )
                        else:
                            print(f"[INFO] Skipping multi-scale comparison for sample {i} (legacy single-scale attention)")
                    else:
                        # Backward compatibility: single attention tensor
                        save_path = os.path.join(attn_dir, f'attention_sample_{i:04d}.png')
                        visualize_attention_weights(
                            attention_weights,
                            tokens=token_ids,
                            caption=caption,
                            save_path=save_path,
                            tokenizer=text_encoder.tokenizer,
                            top_k=min(10, len(caption.split()))
                        )

                except Exception as e:
                    print(f"[WARNING] Failed to visualize attention for sample {i}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                if text_encoder is not None:
                    print(f"[WARNING] No attention weights or token IDs for sample {i}")

        print(f"[INFO] 3-Level attention visualizations saved to: {attn_dir}")

    # Save point clouds as numpy arrays
    if args.save_pointclouds:
        print(f"\n[INFO] Saving point clouds...")
        pc_dir = os.path.join(args.save_dir, 'pointclouds')
        os.makedirs(pc_dir, exist_ok=True)

        for i in range(num_samples):
            np.save(
                os.path.join(pc_dir, f'sample_{i:04d}_gt.npy'),
                all_gt[i].numpy()
            )
            np.save(
                os.path.join(pc_dir, f'sample_{i:04d}_generated.npy'),
                all_generated[i].numpy()
            )

        print(f"[INFO] Point clouds saved to: {pc_dir}")

    print(f"\n[SUCCESS] All visualizations saved to: {args.save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize generated samples vs GT from training captions')

    # Model checkpoint
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to model checkpoint')

    # Dataset
    parser.add_argument('--dataset_path', type=str,
                        default='./data/shapenet_tablechair.hdf5',
                        help='Path to HDF5 dataset')
    parser.add_argument('--categories', type=str, nargs='+', default=['table'],
                        help='Categories to visualize')
    parser.add_argument('--captions_path', type=str,
                        default='./data/captions.tablechair.csv',
                        help='Path to captions CSV file')

    # Sampling
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples to generate and visualize')
    parser.add_argument('--sample_mode', type=str, default='random',
                        choices=['random', 'evenly'],
                        help='How to select samples from dataset')
    parser.add_argument('--custom_captions', type=str, nargs='+', default=None,
                        help='Custom captions to generate from (instead of using dataset)')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode: prompt for captions one by one')
    parser.add_argument('--sample_num_points', type=int, default=1024,
                        help='Number of points to generate')
    parser.add_argument('--flexibility', type=float, default=0.0,
                        help='Flexibility for sampling')

    # Normalization
    parser.add_argument('--normalize', type=str, default='shape_unit',
                        choices=['shape_unit', 'shape_bbox', 'none'],
                        help='Point cloud normalization mode')

    # Output
    parser.add_argument('--save_dir', type=str, default='./visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--samples_per_page', type=int, default=5,
                        help='Number of samples per visualization page')
    parser.add_argument('--save_pointclouds', action='store_true',
                        help='Save point clouds as numpy arrays')
    parser.add_argument('--visualize_attention', action='store_true',
                        help='Visualize attention weights to show which words the model focuses on')

    # Other
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=2020,
                        help='Random seed')

    args = parser.parse_args()

    main(args)
