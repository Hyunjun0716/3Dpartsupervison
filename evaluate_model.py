"""
Comprehensive evaluation script for text-conditioned point cloud generation
Evaluates: CD, EMD, F-score, MMD, FPD, JSD, CLIP-Score, CLIP R-Precision
"""
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.dataset import ShapeNetCoreText
from utils.misc import seed_all
from models.vae_flow import FlowVAE
from models.vae_gaussian import GaussianVAE
from models.clip_encoder import FrozenCLIPTextEmbedder
from evaluation.evaluation_metrics import EMD_CD, _pairwise_EMD_CD_, compute_all_metrics
from evaluation.distribution_metrics import compute_jsd


def load_model(checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    args = checkpoint['args']
    print(f"Model trained on categories: {args.categories}")

    # Check for iteration in different possible locations
    if 'it' in checkpoint:
        print(f"Iteration: {checkpoint['it']}")
    elif 'others' in checkpoint and 'iteration' in checkpoint['others']:
        print(f"Iteration: {checkpoint['others']['iteration']}")
    else:
        print(f"Iteration: Unknown")

    # Build model
    if args.model == 'flow':
        model = FlowVAE(args).to(device)
    else:
        model = GaussianVAE(args).to(device)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Load text encoder if needed
    text_encoder = None
    if args.use_text_condition:
        text_encoder = FrozenCLIPTextEmbedder(
            version=args.clip_model,
            device=device,
            max_length=77,
            return_sequence=True  # Return token sequences for cross-attention
        ).to(device)
        print("CLIP text encoder loaded")

    return model, text_encoder, args


def generate_samples(model, text_encoder, dataset, num_samples, batch_size, device='cuda', model_args=None):
    """Generate point clouds from text prompts"""

    # Use all test samples if num_samples is None or larger than dataset
    if num_samples is None or num_samples > len(dataset):
        num_samples = len(dataset)
        indices = list(range(len(dataset)))
        print(f"\nGenerating samples for ALL {num_samples} test samples...")
    else:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        print(f"\nGenerating {num_samples} samples (sampled from {len(dataset)} test samples)...")

    # Get number of points from model args (training used this)
    num_points = model_args.sample_num_points if model_args else 2048
    print(f"Using {num_points} points (as used in training)")

    generated_samples = []
    ground_truth = []
    captions = []

    for i in tqdm(range(0, len(indices), batch_size), desc="Generating"):
        batch_indices = indices[i:i+batch_size]
        batch_data = [dataset[idx] for idx in batch_indices]

        # Get ground truth and captions
        gt_pcs_full = torch.stack([d['pointcloud'] for d in batch_data]).to(device)
        batch_captions = [d['caption'] for d in batch_data]

        # Downsample ground truth to match training (if needed)
        if gt_pcs_full.shape[1] != num_points:
            # Random subsampling to match training
            indices_subsample = torch.randperm(gt_pcs_full.shape[1])[:num_points]
            gt_pcs = gt_pcs_full[:, indices_subsample, :]
        else:
            gt_pcs = gt_pcs_full

        # Encode text
        if text_encoder is not None:
            with torch.no_grad():
                text_emb = text_encoder(batch_captions)
                text_tokens = text_emb['tokens']  # (B, 77, 512)
        else:
            text_tokens = None

        # Generate samples
        with torch.no_grad():
            # Sample latent code from prior (standard normal)
            batch_size = len(batch_data)
            latent_dim = model.args.latent_dim
            w = torch.randn(batch_size, latent_dim).to(device)

            # FlowVAE.sample expects: sample(w, num_points, flexibility, text_emb, truncate_std)
            # where w is latent code from prior
            gen_pcs = model.sample(
                w=w,
                num_points=num_points,  # Use same number as training
                flexibility=0.0,
                text_emb=text_tokens,
                truncate_std=2.0
            )

        generated_samples.append(gen_pcs.cpu())
        ground_truth.append(gt_pcs.cpu())
        captions.extend(batch_captions)

    generated_samples = torch.cat(generated_samples, dim=0)
    ground_truth = torch.cat(ground_truth, dim=0)

    print(f"Generated: {generated_samples.shape}")
    print(f"Ground truth: {ground_truth.shape}")

    return generated_samples, ground_truth, captions


def compute_geometric_metrics(generated, reference, batch_size=32):
    """Compute CD, EMD, F-score"""
    print("\n" + "="*60)
    print("Computing Geometric Metrics (CD, EMD, F-score)")
    print("="*60)

    # CD and EMD
    results = EMD_CD(generated, reference, batch_size=batch_size, reduced=True)
    cd = results['MMD-CD'].item()
    emd = results['MMD-EMD'].item()

    print(f"Chamfer Distance (CD): {cd:.6f}")
    print(f"Earth Mover's Distance (EMD): {emd:.6f}")

    # F-score at different thresholds
    thresholds = [0.001, 0.002, 0.005]
    fscores = {}

    for thresh in thresholds:
        fscore = compute_fscore(generated, reference, threshold=thresh)
        fscores[f'F-score@{thresh}'] = fscore
        print(f"F-score @ {thresh}: {fscore:.4f}")

    return {
        'CD': cd,
        'EMD': emd,
        **fscores
    }


def compute_fscore(pred, gt, threshold=0.001, batch_size=16):
    """Compute F-score at given threshold (with smaller batch to save memory)"""
    from evaluation.evaluation_metrics import distChamfer

    device = pred.device
    num_samples = pred.shape[0]

    all_precisions = []
    all_recalls = []

    # Process in batches to avoid OOM (reduced batch size for 1024 points)
    for i in range(0, num_samples, batch_size):
        batch_pred = pred[i:i+batch_size]
        batch_gt = gt[i:i+batch_size]

        # Compute distances
        dl, dr = distChamfer(batch_pred, batch_gt)

        # Precision: % of pred points close to gt
        precision = (dl.min(dim=1)[0] < threshold).float().mean()

        # Recall: % of gt points close to pred
        recall = (dr.min(dim=1)[0] < threshold).float().mean()

        all_precisions.append(precision)
        all_recalls.append(recall)

    # Average across batches
    avg_precision = torch.stack(all_precisions).mean()
    avg_recall = torch.stack(all_recalls).mean()

    # F-score
    if avg_precision + avg_recall > 0:
        fscore = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    else:
        fscore = torch.tensor(0.0)

    return fscore.item()


def compute_distribution_metrics(generated, reference, batch_size=32):
    """Compute MMD, FPD, JSD"""
    print("\n" + "="*60)
    print("Computing Distribution Metrics (MMD, FPD, JSD)")
    print("="*60)

    results = {}

    # Use existing compute_all_metrics from evaluation_metrics
    try:
        # compute_all_metrics expects torch tensors
        metrics = compute_all_metrics(
            generated,  # Already torch tensor
            reference,  # Already torch tensor
            batch_size=batch_size
        )
        results.update(metrics)

        print(f"MMD-CD: {metrics.get('lgan_mmd-CD', 'N/A')}")
        print(f"COV-CD: {metrics.get('lgan_cov-CD', 'N/A')}")
        print(f"1-NNA-CD: {metrics.get('1-NN-CD-acc', 'N/A')}")

        # Compute JSD separately
        jsd = compute_jsd(generated, reference)
        results['JSD'] = jsd
        print(f"JSD: {jsd:.6f}")

    except Exception as e:
        print(f"Warning: Could not compute all metrics: {e}")
        import traceback
        traceback.print_exc()
        results['MMD-CD'] = 0.0
        results['JSD'] = 0.0

    return results


def compute_clip_metrics(model, generated, captions, text_encoder, device='cuda'):
    """Compute CLIP-Score and CLIP R-Precision"""
    print("\n" + "="*60)
    print("Computing CLIP Metrics (CLIP-Score, R-Precision)")
    print("="*60)

    if text_encoder is None:
        print("Warning: Text encoder not available, skipping CLIP metrics")
        return {'CLIP-Score': 0.0, 'CLIP-R-Precision': 0.0}

    # Import CLIP metrics functions
    from evaluation.clip_metrics import compute_all_clip_metrics

    try:
        metrics = compute_all_clip_metrics(
            model=model,
            text_encoder=text_encoder,
            point_clouds=generated,
            captions=captions,
            batch_size=32,
            device=device
        )

        print(f"CLIP-Score: {metrics.get('CLIP-Score', 0.0):.4f}")
        print(f"CLIP R-Precision@1: {metrics.get('R-Precision@1', 0.0):.4f}")
        print(f"CLIP R-Precision@5: {metrics.get('R-Precision@5', 0.0):.4f}")
        print(f"CLIP R-Precision@10: {metrics.get('R-Precision@10', 0.0):.4f}")

        return metrics

    except Exception as e:
        print(f"Warning: Could not compute CLIP metrics: {e}")
        import traceback
        traceback.print_exc()
        return {
            'CLIP-Score': 0.0,
            'CLIP-R-Precision@1': 0.0,
            'CLIP-R-Precision@5': 0.0,
            'CLIP-R-Precision@10': 0.0
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                       default='/home/jun/diffusion-point-cloud/logs_gen/GEN_2025_11_30__01_59_45/ckpt_0.000000_500000.pt',
                       help='Path to checkpoint')
    parser.add_argument('--dataset_path', type=str,
                       default='./data/shapenet_tablechair.hdf5')
    parser.add_argument('--captions_path', type=str,
                       default='./data/captions.tablechair.csv')
    parser.add_argument('--modelid_mapping_path', type=str,
                       default='./data/modelid_mapping_tablechair.json')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to generate for evaluation (None = use all test samples)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--output', type=str, default='./evaluation_results.txt')

    args = parser.parse_args()
    seed_all(args.seed)

    # Load model
    model, text_encoder, model_args = load_model(args.checkpoint, args.device)

    # Load dataset
    print("\nLoading test dataset...")
    test_dataset = ShapeNetCoreText(
        path=args.dataset_path,
        cates=model_args.categories,
        split='test',
        scale_mode=model_args.scale_mode,
        captions_path=args.captions_path,
        modelid_mapping_path=args.modelid_mapping_path
    )
    print(f"Test dataset size: {len(test_dataset)}")

    # Generate samples
    generated, reference, captions = generate_samples(
        model, text_encoder, test_dataset,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device,
        model_args=model_args  # Pass model args to get num_points
    )

    # Move to device for metric computation
    generated = generated.to(args.device)
    reference = reference.to(args.device)

    # Compute all metrics
    results = {}

    # 1. Geometric fidelity
    geom_metrics = compute_geometric_metrics(generated, reference, args.batch_size)
    results.update(geom_metrics)

    # 2. Distribution metrics
    dist_metrics = compute_distribution_metrics(generated, reference, args.batch_size)
    results.update(dist_metrics)

    # 3. CLIP metrics
    clip_metrics = compute_clip_metrics(model, generated, captions, text_encoder, args.device)
    results.update(clip_metrics)

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.6f}")
        else:
            print(f"{key:30s}: {value}")

    # Save results
    with open(args.output, 'w') as f:
        f.write("="*60 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Number of samples: {args.num_samples}\n")
        f.write(f"Categories: {model_args.categories}\n")
        f.write("\n")
        for key, value in results.items():
            f.write(f"{key:30s}: {value}\n")

    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
