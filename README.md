# SAM2 tuning

Visualize training data points

```shell
uv run datapoint.py /Users/piotrswiecik/dev/ives/coronary/datasets/arcade/syntax
```

Run training

```shell
uv run train.py --dataset-root /Users/piotrswiecik/dev/ives/coronary/datasets/arcade --epochs=10
```

Single point inference test

```shell
uv run predict_random.py --weights-path /Users/piotrswiecik/dev/ives/coronary/sam2/workdir/artifacts/trained_models/sam2_arcade_ep2.torch --dataset-root /Users/piotrswiecik/dev/ives/coronary/datasets/arcade
```


## ARCADE annotation format

```json
{
  "images": [
    {
      'coco_url': '',
      'date_captured': 0,
      'file_name': '922.png',
      'flickr_url': '',
      'height': 512,
      'id': 922,
      'license': 0,
      'width': 512
    }
  ],
  "annotations": [
    {
      'area': 442.0,
      'attributes': {'occluded': False},
      'bbox': [341.0, 232.0, 41.0, 119.0],
      'category_id': 8,
      'id': 1,
      'image_id': 922,
      'iscrowd': 0,
      'segmentation': [[382.0, ...]]
    }
  ],
  "categories": [
    {'id': 1, 'name': '1', 'supercategory': ''}
  ]
}
```

Available class labels: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]`

