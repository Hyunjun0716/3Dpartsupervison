"""
Geometric fidelity metrics for point cloud generation
Implements Chamfer Distance, EMD, and F-score
"""
import torch
import numpy as np
from tqdm import tqdm


def chamfer_distance(pred, gt, batch_size=8):
    """
    Compute Chamfer Distance between two point cloud sets

    CD(S1, S2) = 1/|S1| * sum(min||p-q||^2) + 1/|S2| * sum(min||q-p||^2)

    Args:
        pred: (N, num_points, 3) predicted point clouds
        gt: (N, num_points, 3) ground truth point clouds
        batch_size: batch size for computation

    Returns:
        cd: mean Chamfer Distance
        cd_per_sample: (N,) CD for each sample
    """
    from .evaluation_metrics import distChamfer

    if not torch.is_tensor(pred):
        pred = torch.from_numpy(pred).float()
    if not torch.is_tensor(gt):
        gt = torch.from_numpy(gt).float()

    device = pred.device
    N = pred.shape[0]

    cd_list = []

    for i in tqdm(range(0, N, batch_size), desc='Chamfer Distance'):
        batch_pred = pred[i:i+batch_size].to(device)
        batch_gt = gt[i:i+batch_size].to(device)

        # distChamfer returns (dl, dr) where:
        # dl: min distance from pred to gt
        # dr: min distance from gt to pred
        dl, dr = distChamfer(batch_pred, batch_gt)

        # CD = mean(dl) + mean(dr)
        cd_batch = dl.mean(dim=1) + dr.mean(dim=1)
        cd_list.append(cd_batch)

    cd_per_sample = torch.cat(cd_list)
    cd_mean = cd_per_sample.mean()

    return cd_mean.item(), cd_per_sample.cpu().numpy()


def earth_movers_distance(pred, gt, batch_size=8):
    """
    Compute Earth Mover's Distance (Wasserstein Distance)

    Note: This is a placeholder as EMD is computationally expensive
    The actual implementation requires optimization solvers

    Args:
        pred: (N, num_points, 3) predicted point clouds
        gt: (N, num_points, 3) ground truth point clouds
        batch_size: batch size

    Returns:
        emd: mean EMD
    """
    from .evaluation_metrics import emd_approx

    if not torch.is_tensor(pred):
        pred = torch.from_numpy(pred).float()
    if not torch.is_tensor(gt):
        gt = torch.from_numpy(gt).float()

    device = pred.device
    N = pred.shape[0]

    emd_list = []

    for i in range(0, N, batch_size):
        batch_pred = pred[i:i+batch_size].to(device)
        batch_gt = gt[i:i+batch_size].to(device)

        emd_batch = emd_approx(batch_pred, batch_gt)
        emd_list.append(emd_batch)

    emd = torch.cat(emd_list).mean()

    return emd.item()


def f_score(pred, gt, threshold=0.01):
    """
    Compute F-score at given threshold

    Precision: % of predicted points within threshold of GT
    Recall: % of GT points within threshold of prediction
    F-score: harmonic mean of precision and recall

    Args:
        pred: (N, num_points, 3) predicted point clouds
        gt: (N, num_points, 3) ground truth point clouds
        threshold: distance threshold

    Returns:
        fscore: F-score
        precision: Precision
        recall: Recall
    """
    from .evaluation_metrics import distChamfer

    if not torch.is_tensor(pred):
        pred = torch.from_numpy(pred).float()
    if not torch.is_tensor(gt):
        gt = torch.from_numpy(gt).float()

    device = pred.device

    # Compute distances
    dl, dr = distChamfer(pred.to(device), gt.to(device))

    # Precision: % of predicted points close to GT
    # For each point in pred, find min distance to gt
    precision = (dl.min(dim=1)[0] < threshold).float().mean()

    # Recall: % of GT points close to prediction
    # For each point in gt, find min distance to pred
    recall = (dr.min(dim=1)[0] < threshold).float().mean()

    # F-score
    if precision + recall > 0:
        fscore = 2 * precision * recall / (precision + recall)
    else:
        fscore = torch.tensor(0.0)

    return fscore.item(), precision.item(), recall.item()


def compute_all_geometric_metrics(pred, gt, batch_size=8):
    """
    Compute all geometric fidelity metrics

    Args:
        pred: (N, num_points, 3) predicted point clouds
        gt: (N, num_points, 3) ground truth point clouds
        batch_size: batch size

    Returns:
        dict of metrics
    """
    metrics = {}

    print("\nComputing Chamfer Distance...")
    cd_mean, cd_per_sample = chamfer_distance(pred, gt, batch_size)
    metrics['CD'] = cd_mean
    metrics['CD-std'] = cd_per_sample.std()

    print("Computing Earth Mover's Distance...")
    emd = earth_movers_distance(pred, gt, batch_size)
    metrics['EMD'] = emd

    print("Computing F-scores...")
    thresholds = [0.001, 0.002, 0.005, 0.01]
    for thresh in thresholds:
        fscore, precision, recall = f_score(pred, gt, threshold=thresh)
        metrics[f'F-score@{thresh}'] = fscore
        metrics[f'Precision@{thresh}'] = precision
        metrics[f'Recall@{thresh}'] = recall

    return metrics
