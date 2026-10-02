# Data preparation

## Required files

| Default file | Contents |
| --- | --- |
| `data/shapenet_tablechair.hdf5` | Groups `03001627` (chair) and `04379243` (table), each containing `train`, `val` and `test` arrays of shape `(num_shapes, num_points, 3)`. |
| `data/captions.tablechair.csv` | Columns `modelId` and `description`. Multiple captions may share a model ID. |
| `data/modelid_mapping_tablechair.json` | Keys such as `03001627_train`, with `available_model_ids` aligned with the HDF5 row order. |

Example mapping entry:

```json
{
  "03001627_train": {
    "available_model_ids": ["model_id_1", "model_id_2"],
    "count": 2
  }
}
```

`ShapeNetCoreText` filters to shapes whose mapped model IDs exist in the caption dictionary. An absent or mismatched mapping can leave the filtered dataset empty. Keep IDs aligned with successful stored point-cloud rows.

## Preprocess chair/table point clouds

The helper supports chair `.pts` files under `<data_dir>/03001627/points/` (or `chair_part/03001627/points/`) and table `.txt` files under `<data_dir>/04379243/`. It writes points, not segmentation labels.

```bash
python -m scripts.prepare_data \
    --data_dir /path/to/raw_data \
    --output data/shapenet_tablechair.hdf5 \
    --mapping_output data/modelid_mapping_tablechair.json \
    --num_points 2048
```

Inputs are sorted before the seeded 80/10/10 split. Model-ID mappings include only successfully stored shapes. This helper's newly created split should not be assumed identical to the published evaluation split.

The current training loop downsamples to 1,024 points. Use the paper's original split, point count and normalization when reproducing published results.

A separate general ShapeNet helper is available:

```bash
python -m scripts.build_shapenet --help
```

Check caption matching:

```bash
python -m scripts.check_dataset
```

External data files are ignored by Git. Captions and original data must be supplied separately.
