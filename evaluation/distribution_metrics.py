"""
Distribution-based metrics for point cloud generation
Implements FPD (Frechet Pointcloud Distance) and JSD (Jensen-Shannon Divergence)
"""
import torch
import numpy as np
from scipy.stats import entropy
from scipy.linalg import sqrtm
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Compute Frechet Distance between two Gaussian distributions

    FD = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))

    Args:
        mu1, mu2: mean vectors
        sigma1, sigma2: covariance matrices
        eps: small value for numerical stability

    Returns:
        frechet_distance: scalar
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print("fid calculation produces singular product; adding eps to diagonal of cov estimates")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fd = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    return fd


def compute_fpd(sample_features, ref_features):
    """
    Compute Frechet Pointcloud Distance (FPD)

    Args:
        sample_features: (N, D) generated point cloud features
        ref_features: (M, D) reference point cloud features

    Returns:
        fpd: Frechet Pointcloud Distance
    """
    # Convert to numpy
    if torch.is_tensor(sample_features):
        sample_features = sample_features.cpu().numpy()
    if torch.is_tensor(ref_features):
        ref_features = ref_features.cpu().numpy()

    # Compute statistics
    mu_sample = np.mean(sample_features, axis=0)
    sigma_sample = np.cov(sample_features, rowvar=False)

    mu_ref = np.mean(ref_features, axis=0)
    sigma_ref = np.cov(ref_features, rowvar=False)

    # Compute FPD
    fpd = compute_frechet_distance(mu_sample, sigma_sample, mu_ref, sigma_ref)

    return fpd


def compute_jsd(sample_pcs, ref_pcs, num_bins=50):
    """
    Compute Jensen-Shannon Divergence between two point cloud distributions

    Args:
        sample_pcs: (N, num_points, 3) generated point clouds
        ref_pcs: (M, num_points, 3) reference point clouds
        num_bins: number of bins for histogram

    Returns:
        jsd: Jensen-Shannon Divergence
    """
    # Flatten point clouds
    if torch.is_tensor(sample_pcs):
        sample_pcs = sample_pcs.cpu().numpy()
    if torch.is_tensor(ref_pcs):
        ref_pcs = ref_pcs.cpu().numpy()

    sample_flat = sample_pcs.reshape(-1)
    ref_flat = ref_pcs.reshape(-1)

    # Compute histograms
    min_val = min(sample_flat.min(), ref_flat.min())
    max_val = max(sample_flat.max(), ref_flat.max())

    hist_sample, _ = np.histogram(sample_flat, bins=num_bins, range=(min_val, max_val), density=True)
    hist_ref, _ = np.histogram(ref_flat, bins=num_bins, range=(min_val, max_val), density=True)

    # Normalize to probability distributions
    hist_sample = hist_sample / (hist_sample.sum() + 1e-10)
    hist_ref = hist_ref / (hist_ref.sum() + 1e-10)

    # Compute JSD
    # JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5*(P+Q)
    m = 0.5 * (hist_sample + hist_ref)

    # Add small epsilon to avoid log(0)
    hist_sample = hist_sample + 1e-10
    hist_ref = hist_ref + 1e-10
    m = m + 1e-10

    jsd = 0.5 * entropy(hist_sample, m) + 0.5 * entropy(hist_ref, m)

    return jsd


def compute_mmd(sample_pcs, ref_pcs, batch_size=100):
    """
    Compute Minimum Matching Distance (MMD)

    Args:
        sample_pcs: (N, num_points, 3) generated point clouds
        ref_pcs: (N, num_points, 3) reference point clouds (same N)

    Returns:
        mmd_cd: MMD using Chamfer Distance
    """
    from .evaluation_metrics import distChamfer

    if not torch.is_tensor(sample_pcs):
        sample_pcs = torch.from_numpy(sample_pcs).float()
    if not torch.is_tensor(ref_pcs):
        ref_pcs = torch.from_numpy(ref_pcs).float()

    device = sample_pcs.device
    N = sample_pcs.shape[0]

    cd_list = []

    for i in range(0, N, batch_size):
        batch_sample = sample_pcs[i:i+batch_size].to(device)
        batch_ref = ref_pcs[i:i+batch_size].to(device)

        dl, dr = distChamfer(batch_sample, batch_ref)
        cd = dl.mean(dim=1) + dr.mean(dim=1)
        cd_list.append(cd)

    mmd_cd = torch.cat(cd_list).mean()

    return mmd_cd.item()


def compute_coverage(sample_features, ref_features, k=5):
    """
    Compute Coverage (COV): percentage of reference samples matched

    Args:
        sample_features: (N, D) generated features
        ref_features: (M, D) reference features
        k: number of nearest neighbors

    Returns:
        coverage: percentage (0-100)
    """
    if torch.is_tensor(sample_features):
        sample_features = sample_features.cpu().numpy()
    if torch.is_tensor(ref_features):
        ref_features = ref_features.cpu().numpy()

    # For each reference, find k nearest samples
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(sample_features)
    distances, indices = nbrs.kneighbors(ref_features)

    # Count how many references have at least one close sample
    # (threshold can be adjusted)
    matched = np.unique(indices.flatten())
    coverage = len(matched) / len(ref_features) * 100

    return coverage


def compute_1nn_accuracy(sample_features, ref_features):
    """
    Compute 1-NN accuracy: test if samples and references can be distinguished
    Lower is better (means distributions are more similar)

    Args:
        sample_features: (N, D) generated features
        ref_features: (N, D) reference features

    Returns:
        accuracy: 1-NN classification accuracy
    """
    if torch.is_tensor(sample_features):
        sample_features = sample_features.cpu().numpy()
    if torch.is_tensor(ref_features):
        ref_features = ref_features.cpu().numpy()

    N = len(sample_features)

    # Combine and create labels
    all_features = np.vstack([sample_features, ref_features])
    all_labels = np.hstack([np.zeros(N), np.ones(N)])

    # Leave-one-out 1-NN classification
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(all_features)
    distances, indices = nbrs.kneighbors(all_features)

    # indices[:, 0] is the point itself, indices[:, 1] is nearest neighbor
    pred_labels = all_labels[indices[:, 1]]

    accuracy = (pred_labels == all_labels).mean()

    return accuracy


def compute_all_distribution_metrics(
    sample_pcs,
    ref_pcs,
    sample_features=None,
    ref_features=None,
    batch_size=8
):
    """
    Compute all distribution-based metrics

    Args:
        sample_pcs: (N, num_points, 3) generated point clouds
        ref_pcs: (N, num_points, 3) reference point clouds
        sample_features: (N, D) optional precomputed features
        ref_features: (N, D) optional precomputed features
        batch_size: batch size

    Returns:
        dict of metrics
    """
    metrics = {}

    print("\nComputing MMD...")
    mmd = compute_mmd(sample_pcs, ref_pcs, batch_size)
    metrics['MMD-CD'] = mmd

    print("Computing JSD...")
    jsd = compute_jsd(sample_pcs, ref_pcs)
    metrics['JSD'] = jsd

    if sample_features is not None and ref_features is not None:
        print("Computing FPD...")
        fpd = compute_fpd(sample_features, ref_features)
        metrics['FPD'] = fpd

        print("Computing Coverage...")
        cov = compute_coverage(sample_features, ref_features)
        metrics['Coverage'] = cov

        print("Computing 1-NN Accuracy...")
        nn_acc = compute_1nn_accuracy(sample_features, ref_features)
        metrics['1-NN-Accuracy'] = nn_acc

    return metrics
