"""
Create HDF5 file for chair and table categories combined
"""

from utils.paths import project_path
import argparse
from pathlib import Path
import os
import json

DATA_DIR = Path(project_path('data'))

# Category mappings
synsetid_to_cate = {
    '03001627': 'chair',
    '04379243': 'table',
}

def load_points_and_labels(category_id, model_id, num_points=2048):
    """Load point cloud and part labels for a given model"""
    # Try .pts format (chair)
    base_path = str(DATA_DIR / f'{category_id}/points/{model_id}.pts')
    label_path = str(DATA_DIR / f'{category_id}/points_label/{model_id}.seg')

    if not os.path.exists(base_path):
        # Try alternative path for chair
        base_path = str(DATA_DIR / f'chair_part/{category_id}/points/{model_id}.pts')
        label_path = str(DATA_DIR / f'chair_part/{category_id}/points_label/{model_id}.seg')

    if not os.path.exists(base_path):
        # Try .txt format (table)
        base_path = str(DATA_DIR / f'{category_id}/{model_id}.txt')

    if not os.path.exists(base_path):
        return None, None

    # Load point cloud
    points = np.loadtxt(base_path, dtype=np.float32)

    # If points have more than 3 columns, extract only the first 3 (x, y, z coordinates)
    if points.ndim == 2 and points.shape[1] > 3:
        points = points[:, :3]

    # Load labels if available
    labels = None
    if os.path.exists(label_path):
        labels = np.loadtxt(label_path, dtype=np.int32)

    return points, labels

def create_train_val_test_split(model_ids, train_ratio=0.8, val_ratio=0.1):
    """Split model IDs into train/val/test sets"""
    np.random.seed(2020)
    n_models = len(model_ids)

    # Shuffle
    shuffled_ids = model_ids.copy()
    np.random.shuffle(shuffled_ids)

    # Split
    n_train = int(n_models * train_ratio)
    n_val = int(n_models * val_ratio)

    train_ids = shuffled_ids[:n_train]
    val_ids = shuffled_ids[n_train:n_train+n_val]
    test_ids = shuffled_ids[n_train+n_val:]

    return train_ids, val_ids, test_ids

def process_category(category_id, category_name, num_points=2048):
    """Process all models for a category and split into train/val/test"""
    print(f"\nProcessing {category_name} (ID: {category_id})...")

    # Find points directory and files
    points_dir = str(DATA_DIR / f'{category_id}/points')
    file_extension = '.pts'

    if not os.path.exists(points_dir):
        points_dir = str(DATA_DIR / f'chair_part/{category_id}/points')

    if not os.path.exists(points_dir):
        # Try .txt format (table)
        points_dir = str(DATA_DIR / f'{category_id}')
        file_extension = '.txt'

    if not os.path.exists(points_dir):
        print(f"Warning: Points directory not found for {category_name}")
        return {}, {}

    # Get all model IDs
    model_files = sorted(f for f in os.listdir(points_dir) if f.endswith(file_extension))
    model_ids = [f[:f.rfind(file_extension)] for f in model_files]  # Remove extension

    print(f"Found {len(model_ids)} models for {category_name}")

    # Split into train/val/test
    train_ids, val_ids, test_ids = create_train_val_test_split(model_ids)

    splits_data = {
        'train': [],
        'val': [],
        'test': []
    }

    split_model_ids = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }

    stored_model_ids = {name: [] for name in splits_data}

    for split_name, split_ids in split_model_ids.items():
        print(f"Processing {split_name} split ({len(split_ids)} models)...")

        for model_id in tqdm(split_ids, desc=f"{category_name} {split_name}"):
            points, labels = load_points_and_labels(category_id, model_id, num_points)

            if points is None:
                continue

            # Resample to target number of points
            n_points = points.shape[0]
            if n_points >= num_points:
                # Random sampling
                indices = np.random.choice(n_points, num_points, replace=False)
            else:
                # Oversample with replacement
                indices = np.random.choice(n_points, num_points, replace=True)

            sampled_points = points[indices]
            splits_data[split_name].append(sampled_points)
            stored_model_ids[split_name].append(model_id)

    # Convert to numpy arrays
    for split_name in splits_data:
        if len(splits_data[split_name]) > 0:
            splits_data[split_name] = np.stack(splits_data[split_name], axis=0)
            print(f"{category_name} {split_name}: {splits_data[split_name].shape}")
        else:
            splits_data[split_name] = np.empty((0, num_points, 3), dtype=np.float32)

    # Prepare model ID mapping for this category
    model_id_mapping = {}
    for split_name, split_ids in stored_model_ids.items():
        # split_ids is already a list from numpy array conversion
        if isinstance(split_ids, np.ndarray):
            split_ids_list = split_ids.tolist()
        else:
            split_ids_list = split_ids
        model_id_mapping[f"{category_id}_{split_name}"] = {
            'available_model_ids': split_ids_list,
            'count': len(split_ids_list)
        }

    return splits_data, model_id_mapping

def main():
    global DATA_DIR, h5py, np, tqdm
    parser = argparse.ArgumentParser(description='Prepare matched chair/table point clouds')
    parser.add_argument('--data_dir', default=project_path('data'))
    parser.add_argument('--output', default=project_path('data/shapenet_tablechair.hdf5'))
    parser.add_argument('--mapping_output', default=project_path('data/modelid_mapping_tablechair.json'))
    parser.add_argument('--num_points', type=int, default=2048)
    args = parser.parse_args()
    import h5py
    import numpy as np
    from tqdm import tqdm
    DATA_DIR = Path(args.data_dir)
    output_path = args.output
    mapping_path = args.mapping_output
    num_points = args.num_points
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(mapping_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Creating HDF5 file: {output_path}")
    print(f"Number of points per model: {num_points}")

    # Create HDF5 file
    with h5py.File(output_path, 'w') as f:
        all_model_id_mapping = {}

        for category_id, category_name in synsetid_to_cate.items():
            splits_data, model_id_mapping = process_category(category_id, category_name, num_points)

            # Create group for this category
            cat_group = f.create_group(category_id)

            # Store train/val/test splits
            for split_name, data in splits_data.items():
                cat_group.create_dataset(split_name, data=data, compression='gzip')
                print(f"Saved {category_id}/{split_name}: {data.shape}")

            # Merge model ID mapping
            all_model_id_mapping.update(model_id_mapping)

    # Save model ID mapping
    with open(mapping_path, 'w') as f:
        json.dump(all_model_id_mapping, f, indent=2)

    print(f"\nHDF5 file created: {output_path}")
    print(f"Model ID mapping saved: {mapping_path}")

    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    with h5py.File(output_path, 'r') as f:
        for category_id, category_name in synsetid_to_cate.items():
            if category_id in f:
                print(f"\n{category_name} ({category_id}):")
                for split_name in ['train', 'val', 'test']:
                    if split_name in f[category_id]:
                        shape = f[category_id][split_name].shape
                        print(f"  {split_name}: {shape[0]} models, {shape[1]} points each")

if __name__ == '__main__':
    main()
