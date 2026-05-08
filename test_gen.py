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
from models.clip_encoder import FrozenCLIPTextEmbedder
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
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--batch_size', type=int, default=64)
# Sampling
parser.add_argument('--sample_num_points', type=int, default=1024)
parser.add_argument('--normalize', type=str, default='shape_unit', choices=[None, 'shape_unit', 'shape_bbox'])
parser.add_argument('--seed', type=int, default=9988)
# Text conditioning
parser.add_argument('--use_text_condition', type=eval, default=True, choices=[True, False])
parser.add_argument('--text_prompt', type=str, default=None, help='Text prompt for conditional generation (if None, uses default)')
parser.add_argument('--captions_path', type=str, default='/home/jun/diffusion-point-cloud/data/chairs_only.csv', help='Path to captions CSV file')
parser.add_argument('--clip_model', type=str, default='openai/clip-vit-base-patch32', help='CLIP model version')
args = parser.parse_args()


# Logging
save_dir = os.path.join(args.save_dir, 'GEN_Ours_%s_%d' % ('_'.join(args.categories), int(time.time())) )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logger = get_logger('test', save_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

# Checkpoint
ckpt = torch.load(args.ckpt, weights_only=False)
seed_all(args.seed)

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

# Text encoder (CLIP) - frozen (initialize BEFORE model for stop word filtering)
text_encoder = None
if args.use_text_condition:
    logger.info('Loading CLIP text encoder...')
    text_encoder = FrozenCLIPTextEmbedder(
        version=args.clip_model,
        device=args.device,
        max_length=77,
        return_sequence=True  # Return token sequence for cross attention
    )
    text_encoder = text_encoder.to(args.device)
    logger.info('CLIP text encoder loaded and frozen (returning token sequences for cross attention).')

# Model
logger.info('Loading model...')
if ckpt['args'].model == 'gaussian':
    model = GaussianVAE(ckpt['args'], tokenizer=text_encoder.tokenizer if text_encoder else None).to(args.device)
elif ckpt['args'].model == 'flow':
    model = FlowVAE(ckpt['args'], tokenizer=text_encoder.tokenizer if text_encoder else None).to(args.device)
logger.info(repr(model))
# if ckpt['args'].spectral_norm:
#     add_spectral_norm(model, logger=logger)
model.load_state_dict(ckpt['state_dict'])

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
logger.info(f'Sample captions: {ref_captions[:3]}')  # Show first 3 captions

# Downsample GT point clouds to match generation (1024 points)
# This ensures fair comparison with generated point clouds
if ref_pcs.shape[1] != args.sample_num_points:
    logger.info(f'Downsampling GT from {ref_pcs.shape[1]} to {args.sample_num_points} points for fair comparison')
    # Random sampling without replacement
    indices = torch.randperm(ref_pcs.shape[1])[:args.sample_num_points]
    ref_pcs = ref_pcs[:, indices, :]
    logger.info(f'GT point clouds shape after downsampling: {ref_pcs.shape}')

# Generate Point Clouds using actual captions
logger.info('Generating point clouds using actual captions from dataset...')
gen_pcs = []

for batch_idx in tqdm(range(0, len(test_dset), args.batch_size), desc='Generate'):
    batch_end = min(batch_idx + args.batch_size, len(test_dset))
    batch_size_actual = batch_end - batch_idx

    with torch.no_grad():
        z = torch.randn([batch_size_actual, ckpt['args'].latent_dim]).to(args.device)

        # Get captions for this batch
        batch_captions = ref_captions[batch_idx:batch_end]

        # Encode captions with CLIP
        batch_text_emb = None
        if args.use_text_condition and text_encoder is not None:
            if args.text_prompt:
                # Override: use single prompt for all if specified
                batch_captions = [args.text_prompt] * batch_size_actual
                logger.info(f'Using override prompt: "{args.text_prompt}"')

            batch_text_emb = text_encoder(batch_captions)

        x = model.sample(z, args.sample_num_points, flexibility=ckpt['args'].flexibility, text_emb=batch_text_emb)
        gen_pcs.append(x.detach().cpu())

gen_pcs = torch.cat(gen_pcs, dim=0)
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
logger.info('EVALUATION RESULTS')
logger.info('='*50)
for k, v in results.items():
    logger.info('%s: %.12f' % (k, v))
logger.info('='*50)
