"""
Test loading the table+chair dataset with captions
"""

from utils.paths import project_path
import torch
from utils.dataset import ShapeNetCoreText

# Test dataset loading
print("Testing table+chair dataset loading...")
print("="*60)

dataset = ShapeNetCoreText(
    path=project_path('data/shapenet_tablechair.hdf5'),
    cates=['chair', 'table'],
    split='train',
    scale_mode='shape_unit',
    captions_path=project_path('data/captions.tablechair.csv'),
    modelid_mapping_path=project_path('data/modelid_mapping_tablechair.json'),
)

print(f"\nDataset loaded successfully!")
print(f"Total samples: {len(dataset)}")

# Test sampling a few examples
print("\n" + "="*60)
print("Sample data:")
print("="*60)

for i in range(min(5, len(dataset))):
    sample = dataset[i]
    print(f"\nSample {i+1}:")
    print(f"  Category: {sample['cate']}")
    print(f"  Point cloud shape: {sample['pointcloud'].shape}")
    print(f"  Caption: {sample['caption'][:80]}..." if len(sample['caption']) > 80 else f"  Caption: {sample['caption']}")
    print(f"  Caption matched: {sample['caption_matched']}")
    print(f"  Model ID: {sample.get('model_id', 'N/A')}")

# Count categories (sample 100 items for speed)
sample_size = min(100, len(dataset))
chair_count = 0
table_count = 0
matched_count = 0

import random
random.seed(2020)
sample_indices = random.sample(range(len(dataset)), sample_size)

for i in sample_indices:
    sample = dataset[i]
    if sample['cate'] == 'chair':
        chair_count += 1
    elif sample['cate'] == 'table':
        table_count += 1
    if sample['caption_matched']:
        matched_count += 1

print("\n" + "="*60)
print(f"Category distribution (sampled {sample_size} items):")
print(f"  Chairs: {chair_count}")
print(f"  Tables: {table_count}")
print(f"  Total: {chair_count + table_count}")

# Test caption matching rate
print(f"\nCaption match rate (sampled): {matched_count}/{sample_size} ({100*matched_count/sample_size:.1f}%)")

print("\n" + "="*60)
print("Test completed successfully!")
print("="*60)
