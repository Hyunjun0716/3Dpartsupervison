"""
CLIP-based metrics for text-to-3D evaluation
Implements CLIP-Score and CLIP R-Precision
"""
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def compute_clip_score(point_features, text_features, reduction='mean'):
    """
    Compute CLIP-Score: cosine similarity between point cloud and text features

    Args:
        point_features: (N, D) point cloud features from CLIP-like encoder
        text_features: (N, D) text features from CLIP text encoder
        reduction: 'mean' or 'none'

    Returns:
        clip_score: scalar or (N,) tensor of cosine similarities
    """
    # Normalize features
    point_features = F.normalize(point_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # Compute cosine similarity
    similarity = (point_features * text_features).sum(dim=-1)

    if reduction == 'mean':
        return similarity.mean()
    else:
        return similarity


def compute_clip_r_precision(point_features, text_features, R=1):
    """
    Compute CLIP R-Precision: retrieval accuracy
    For each point cloud, check if the correct text is in top-R retrievals

    Args:
        point_features: (N, D) point cloud features
        text_features: (N, D) text features (same order as point clouds)
        R: top-R retrievals to consider (default=1 for standard R-Precision)

    Returns:
        r_precision: retrieval accuracy (0-1)
    """
    N = point_features.shape[0]

    # Normalize features
    point_features = F.normalize(point_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # Compute similarity matrix (N x N)
    # similarity[i, j] = similarity between point_i and text_j
    similarity = point_features @ text_features.T  # (N, N)

    # For each point cloud, find top-R most similar texts
    top_indices = similarity.topk(k=R, dim=1)[1]  # (N, R)

    # Check if correct text (diagonal) is in top-R
    correct = torch.arange(N, device=point_features.device).unsqueeze(1)  # (N, 1)
    matches = (top_indices == correct).any(dim=1)  # (N,)

    r_precision = matches.float().mean()

    return r_precision.item()


def compute_clip_retrieval_metrics(point_features, text_features):
    """
    Compute comprehensive retrieval metrics:
    - R-Precision @ R=1, 5, 10
    - Median Rank
    - Mean Reciprocal Rank (MRR)

    Args:
        point_features: (N, D) point cloud features
        text_features: (N, D) text features

    Returns:
        dict of metrics
    """
    N = point_features.shape[0]

    # Normalize features
    point_features = F.normalize(point_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # Compute similarity matrix
    similarity = point_features @ text_features.T  # (N, N)

    # Get rankings (for each point cloud, rank all texts)
    ranks = similarity.argsort(dim=1, descending=True)  # (N, N)

    # Find rank of correct text (diagonal indices)
    correct_indices = torch.arange(N, device=point_features.device)
    correct_ranks = (ranks == correct_indices.unsqueeze(1)).nonzero()[:, 1]  # (N,)

    # Convert to 1-indexed ranks
    correct_ranks = correct_ranks + 1

    # Compute metrics
    metrics = {
        'R-Precision@1': (correct_ranks <= 1).float().mean().item(),
        'R-Precision@5': (correct_ranks <= 5).float().mean().item(),
        'R-Precision@10': (correct_ranks <= 10).float().mean().item(),
        'Median-Rank': correct_ranks.float().median().item(),
        'MRR': (1.0 / correct_ranks.float()).mean().item(),
    }

    return metrics


def encode_point_clouds_with_model(model, point_clouds, batch_size=8, device='cuda'):
    """
    Encode point clouds using the VAE encoder to get features

    Args:
        model: trained VAE model
        point_clouds: (N, num_points, 3) tensor
        batch_size: batch size for encoding
        device: cuda or cpu

    Returns:
        features: (N, latent_dim) encoded features
    """
    model.eval()
    features_list = []

    with torch.no_grad():
        for i in range(0, len(point_clouds), batch_size):
            batch = point_clouds[i:i+batch_size].to(device)

            # Get latent code from VAE encoder
            # Assumes model has encode method that returns mean
            z_mu, z_logvar = model.encoder(batch)

            features_list.append(z_mu.cpu())

    features = torch.cat(features_list, dim=0)
    return features


def compute_all_clip_metrics(
    model,
    text_encoder,
    point_clouds,
    captions,
    batch_size=8,
    device='cuda'
):
    """
    Compute all CLIP-based metrics

    Args:
        model: trained VAE model for encoding point clouds
        text_encoder: CLIP text encoder
        point_clouds: (N, num_points, 3) generated point clouds
        captions: list of N text captions
        batch_size: batch size
        device: cuda or cpu

    Returns:
        dict of all CLIP metrics
    """
    print("Encoding point clouds...")
    point_features = encode_point_clouds_with_model(
        model, point_clouds, batch_size, device
    ).to(device)

    print("Encoding text captions...")
    text_features_list = []
    for i in tqdm(range(0, len(captions), batch_size), desc="Text encoding"):
        batch_captions = captions[i:i+batch_size]
        with torch.no_grad():
            text_emb = text_encoder(batch_captions)
            # Use pooled features for similarity
            text_features_list.append(text_emb['pool'].cpu())

    text_features = torch.cat(text_features_list, dim=0).to(device)

    # Project point features to same dimension as text features if needed
    point_dim = point_features.shape[-1]
    text_dim = text_features.shape[-1]

    if point_dim != text_dim:
        print(f"Projecting point features from {point_dim}D to {text_dim}D...")
        # Simple linear projection
        projection = torch.nn.Linear(point_dim, text_dim).to(device)
        with torch.no_grad():
            point_features = projection(point_features)

    # Compute metrics
    metrics = {}

    # CLIP-Score
    clip_score = compute_clip_score(point_features, text_features)
    metrics['CLIP-Score'] = clip_score.item()

    # Retrieval metrics
    retrieval_metrics = compute_clip_retrieval_metrics(point_features, text_features)
    metrics.update(retrieval_metrics)

    return metrics
