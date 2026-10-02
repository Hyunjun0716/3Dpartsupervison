"""
Test if text conditioning is working properly
Generate samples with specific captions and check if they differ
"""

from utils.paths import project_path
import argparse
import torch
import numpy as np
from models.vae_flow import FlowVAE
from models.clip_encoder import FrozenCLIPTextEmbedder

# Load model
parser = argparse.ArgumentParser(description='Legacy text-conditioning diagnostic')
parser.add_argument('--checkpoint', required=True)
checkpoint_path = parser.parse_args().checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
args = checkpoint['args']

model = FlowVAE(args).cuda()
model.load_state_dict(checkpoint['state_dict'])
model.eval()

text_encoder = FrozenCLIPTextEmbedder(
    version=args.clip_model,
    device='cuda',
    max_length=77,
    return_sequence=True
).cuda()

# Test captions
test_captions = [
    "a wooden chair with four legs",
    "a round dining table",
    "an office chair with wheels",
    "a small coffee table",
]

print("Testing text conditioning...")
print("=" * 60)

# Generate with same latent but different text
torch.manual_seed(42)
w = torch.randn(len(test_captions), args.latent_dim).cuda()

for i, caption in enumerate(test_captions):
    print(f"\n{i+1}. Caption: '{caption}'")

    # Encode text
    with torch.no_grad():
        text_emb = text_encoder([caption])
        text_tokens = text_emb['tokens']

        # Generate with this caption
        gen_pc = model.sample(
            w=w[i:i+1],
            num_points=args.sample_num_points,
            flexibility=0.0,
            text_emb=text_tokens,
            truncate_std=2.0
        )

        # Basic statistics
        print(f"   Generated shape: {gen_pc.shape}")
        print(f"   Mean: {gen_pc.mean(dim=(1,2)).item():.4f}")
        print(f"   Std: {gen_pc.std(dim=(1,2)).item():.4f}")
        print(f"   Min: {gen_pc.min().item():.4f}, Max: {gen_pc.max().item():.4f}")

print("\n" + "=" * 60)
print("Now generating with SAME latent, different texts:")
print("If text conditioning works, outputs should differ significantly")

# Use SAME latent for all
w_same = torch.randn(1, args.latent_dim).cuda()
w_same = w_same.repeat(len(test_captions), 1)

all_samples = []
for i, caption in enumerate(test_captions):
    with torch.no_grad():
        text_emb = text_encoder([caption])
        text_tokens = text_emb['tokens']

        gen_pc = model.sample(
            w=w_same[i:i+1],
            num_points=args.sample_num_points,
            flexibility=0.0,
            text_emb=text_tokens,
            truncate_std=2.0
        )
        all_samples.append(gen_pc)

all_samples = torch.cat(all_samples, dim=0)

print(f"\nGenerated {len(all_samples)} samples with SAME latent")
print("Pairwise Chamfer Distances (should be SMALL if text has no effect):")

from evaluation.evaluation_metrics import distChamfer

for i in range(len(all_samples)):
    for j in range(i+1, len(all_samples)):
        dl, dr = distChamfer(all_samples[i:i+1], all_samples[j:j+1])
        cd = (dl.mean() + dr.mean()).item()
        print(f"  '{test_captions[i][:30]}...' vs '{test_captions[j][:30]}...': CD = {cd:.4f}")

print("\n" + "=" * 60)
print("Interpretation:")
print("- If CD values are SMALL (< 0.1): Text conditioning NOT working")
print("- If CD values are LARGE (> 0.5): Text conditioning IS working")
