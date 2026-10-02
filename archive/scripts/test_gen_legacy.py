"""
Legacy test script for old checkpoints (without CLIP cross-attention, tokenizer, etc.)
For checkpoints trained with earlier versions that may not have:
- Tokenizer support
- Cross-attention
- Stop word filtering
"""

from utils.paths import project_path
import os
import time
import math
import argparse
import torch
from tqdm.auto import tqdm

from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.vae_gaussian import *
from models.vae_flow import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from evaluation import *

def normalize_point_clouds(pcs, mode, logger):
    if mode is None:
        logger.info('Will not normalize point clouds.')
        return pcs
    logger.info('Normalization mode: %s' % mode)
    for i in tqdm(range(pcs.size(0)), desc='Normalize'):
        pc = pcs[i]
        if mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        pc = (pc - shift) / scale
        pcs[i] = pc
    return pcs


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint file')
parser.add_argument('--categories', type=str_list, default=['chair'], help='Categories to test')
parser.add_argument('--save_dir', type=str, default=project_path('results_legacy'))
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default=project_path('data/shapenet.hdf5'))
parser.add_argument('--batch_size', type=int, default=64)
# Sampling (will be overridden by checkpoint's sample_num_points)
parser.add_argument('--sample_num_points', type=int, default=None, help='Override checkpoint sample_num_points (if None, use checkpoint value)')
parser.add_argument('--normalize', type=str, default='shape_unit', choices=[None, 'shape_unit', 'shape_bbox'])
parser.add_argument('--seed', type=int, default=9988)
# Text conditioning - for legacy checkpoints
parser.add_argument('--use_text_condition', type=eval, default=True, choices=[True, False], help='Use text conditioning (will auto-detect from checkpoint)')
parser.add_argument('--captions_path', type=str, default=project_path('data/chairs_only.csv'), help='Path to captions CSV file')
args = parser.parse_args()


# Logging
save_dir = os.path.join(args.save_dir, 'GEN_Legacy_%s_%d' % ('_'.join(args.categories), int(time.time())) )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logger = get_logger('test_legacy', save_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

# Checkpoint
logger.info(f'Loading checkpoint from {args.ckpt}')
ckpt = torch.load(args.ckpt, weights_only=False)
seed_all(args.seed)

# Log checkpoint info
logger.info('Checkpoint info:')
logger.info(f'  - Model type: {ckpt["args"].model}')
logger.info(f'  - Latent dim: {ckpt["args"].latent_dim}')
logger.info(f'  - Checkpoint sample_num_points: {ckpt["args"].sample_num_points}')
logger.info(f'  - Use text condition: {ckpt["args"].use_text_condition}')
logger.info(f'  - Scale mode: {ckpt["args"].scale_mode}')

# Determine sample_num_points
if args.sample_num_points is None:
    sample_num_points = ckpt['args'].sample_num_points
    logger.info(f'Using checkpoint sample_num_points: {sample_num_points}')
else:
    sample_num_points = args.sample_num_points
    logger.info(f'Overriding sample_num_points to: {sample_num_points}')

# Datasets and loaders
logger.info('Loading datasets...')

# Use ShapeNetCoreText if using text conditioning, otherwise use standard ShapeNetCore
if args.use_text_condition and args.captions_path:
    from utils.dataset import ShapeNetCoreText
    test_dset = ShapeNetCoreText(
        path=args.dataset_path,
        cates=args.categories,
        split='test',
        scale_mode=args.normalize,
        captions_path=args.captions_path,
    )
    logger.info(f'Using ShapeNetCoreText dataset with captions from {args.captions_path}')
else:
    test_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=args.categories,
        split='test',
        scale_mode=args.normalize,
    )
    logger.info('Using standard ShapeNetCore dataset (no text conditioning)')

test_loader = DataLoader(test_dset, batch_size=args.batch_size, num_workers=0)

# Text encoder - ONLY if using text conditioning
# Legacy checkpoints don't have tokenizer, so we pass None
text_encoder = None
if args.use_text_condition:
    try:
        from models.clip_encoder import FrozenCLIPTextEmbedder
        logger.info('Loading CLIP text encoder...')
        text_encoder = FrozenCLIPTextEmbedder(
            version='openai/clip-vit-base-patch32',
            device=args.device,
            max_length=77,
            return_sequence=True
        )
        text_encoder = text_encoder.to(args.device)
        logger.info('CLIP text encoder loaded.')
    except Exception as e:
        logger.warning(f'Failed to load CLIP encoder: {e}')
        logger.warning('Will proceed without text conditioning.')
        args.use_text_condition = False

# Model - Use checkpoint's exact args to recreate the original architecture
logger.info('Loading model with checkpoint args to match original architecture...')

# Use the checkpoint's args directly to ensure architecture compatibility
model_args = ckpt['args']

# Override device to current device
original_device = model_args.device
model_args.device = args.device

try:
    if model_args.model == 'gaussian':
        model = GaussianVAE(model_args, tokenizer=None).to(args.device)
    elif model_args.model == 'flow':
        model = FlowVAE(model_args, tokenizer=None).to(args.device)
    logger.info('Model architecture created (using checkpoint args)')
except Exception as e:
    logger.error(f'Failed to create model: {e}')
    logger.error('This checkpoint may have incompatible architecture')
    raise

# Restore original device in args
model_args.device = original_device

logger.info(repr(model))

# Load state dict with strict=False to handle architecture mismatches
# Legacy checkpoints may have different cross-attention architecture
try:
    model.load_state_dict(ckpt['state_dict'], strict=True)
    logger.info('Model weights loaded successfully (strict mode)')
except RuntimeError as e:
    logger.warning(f'Strict loading failed: {e}')
    logger.warning('Attempting to load with strict=False (architecture mismatch detected)')

    # Load with strict=False
    missing_keys, unexpected_keys = model.load_state_dict(ckpt['state_dict'], strict=False)

    if missing_keys:
        logger.warning(f'Missing keys ({len(missing_keys)}): {missing_keys[:5]}...')
    if unexpected_keys:
        logger.warning(f'Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...')

    logger.info('Model weights loaded with strict=False')

# Reference Point Clouds and collect captions
logger.info(f'Test dataset size: {len(test_dset)}')
ref_pcs = []
ref_captions = []
for i, data in enumerate(test_dset):
    ref_pcs.append(data['pointcloud'].unsqueeze(0))
    # Get caption if available
    if 'caption' in data:
        ref_captions.append(data['caption'])
    else:
        # Fallback to category name if no caption
        ref_captions.append(f"a {data['cate']}")

if len(ref_pcs) == 0:
    logger.error('No test data found! Please check:')
    logger.error(f'  - Dataset path: {args.dataset_path}')
    logger.error(f'  - Categories: {args.categories}')
    logger.error(f'  - Captions path: {args.captions_path}')
    raise ValueError('Test dataset is empty')

ref_pcs = torch.cat(ref_pcs, dim=0)
logger.info(f'Loaded {len(ref_pcs)} reference point clouds')
logger.info(f'Reference point clouds shape: {ref_pcs.shape}')
if args.use_text_condition:
    logger.info(f'Sample captions: {ref_captions[:3]}')

# Downsample GT point clouds to match generation
if ref_pcs.shape[1] != sample_num_points:
    logger.info(f'Downsampling GT from {ref_pcs.shape[1]} to {sample_num_points} points for fair comparison')
    # Random sampling without replacement
    indices = torch.randperm(ref_pcs.shape[1])[:sample_num_points]
    ref_pcs = ref_pcs[:, indices, :]
    logger.info(f'GT point clouds shape after downsampling: {ref_pcs.shape}')

# Generate Point Clouds
if args.use_text_condition and text_encoder is not None:
    logger.info('Generating point clouds using actual captions from dataset...')
else:
    logger.info('Generating point clouds without text conditioning...')

gen_pcs = []

for batch_idx in tqdm(range(0, len(test_dset), args.batch_size), desc='Generate'):
    batch_end = min(batch_idx + args.batch_size, len(test_dset))
    batch_size_actual = batch_end - batch_idx

    with torch.no_grad():
        z = torch.randn([batch_size_actual, ckpt['args'].latent_dim]).to(args.device)

        # Encode captions with CLIP if using text conditioning
        batch_text_emb = None
        if args.use_text_condition and text_encoder is not None:
            batch_captions = ref_captions[batch_idx:batch_end]
            batch_text_emb = text_encoder(batch_captions)
        elif ckpt['args'].use_text_condition:
            # Legacy checkpoint expects text_emb but we don't have real captions
            # Create dummy text embeddings to avoid None error
            logger.warning('Creating dummy text embeddings for legacy checkpoint (first batch only)')
            # Create a dummy caption
            dummy_captions = [f"a {args.categories[0]}"] * batch_size_actual
            if text_encoder is not None:
                batch_text_emb = text_encoder(dummy_captions)
            else:
                # If CLIP encoder failed to load, create random embeddings
                # Legacy models expect dict format with 'tokens' key
                batch_text_emb = {
                    'tokens': torch.randn(batch_size_actual, 77, 512).to(args.device),
                    'pool': torch.randn(batch_size_actual, 512).to(args.device),
                    'token_ids': torch.ones(batch_size_actual, 77).long().to(args.device)
                }

        # Sample
        try:
            x = model.sample(z, sample_num_points, flexibility=ckpt['args'].flexibility, text_emb=batch_text_emb)
        except Exception as e:
            logger.error(f'Sampling failed: {e}')
            logger.error('This checkpoint may be incompatible with current code')
            raise

        gen_pcs.append(x.detach().cpu())

gen_pcs = torch.cat(gen_pcs, dim=0)
logger.info(f'Generated {len(gen_pcs)} point clouds')
logger.info(f'Generated point clouds shape: {gen_pcs.shape}')

if args.normalize is not None:
    gen_pcs = normalize_point_clouds(gen_pcs, mode=args.normalize, logger=logger)

# Save
logger.info('Saving point clouds...')
np.save(os.path.join(save_dir, 'out.npy'), gen_pcs.numpy())

# Compute metrics
logger.info('Computing evaluation metrics...')
logger.info(f'Generated point clouds shape: {gen_pcs.shape}')
logger.info(f'Reference point clouds shape: {ref_pcs.shape}')

with torch.no_grad():
    # Compute CD-based metrics (Coverage, MMD, 1-NN)
    results = compute_all_metrics(gen_pcs.to(args.device), ref_pcs.to(args.device), args.batch_size)
    results = {k:v.item() for k, v in results.items()}

    # Compute JSD
    jsd = jsd_between_point_cloud_sets(gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
    results['jsd'] = jsd

logger.info('\n' + '='*50)
logger.info('EVALUATION RESULTS (LEGACY CHECKPOINT)')
logger.info('='*50)
for k, v in results.items():
    logger.info('%s: %.12f' % (k, v))
logger.info('='*50)
