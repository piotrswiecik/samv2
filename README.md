# SAM2 tuning

Visualize training data points

```shell
uv run datapoint.py --dataset-root /Users/piotrswiecik/dev/ives/coronary/datasets/arcade/
```

Run training for SAM

```shell
uv run train.py --dataset-root /Users/piotrswiecik/dev/ives/coronary/datasets/arcade --epochs=100
uv run train.py --dataset-root /home/ives/piotr/arcade --epochs=100
```

Run training for ResNet

```shell
# prepare classifier data
uv run classifier_data.py --dataset-root=/Users/piotrswiecik/dev/ives/coronary/datasets/arcade --out-dir=/Users/piotrswiecik/dev/ives/coronary/sam2/workdir/arcade_classifier_data

# run
uv run resnet.py --dataset-root /Users/piotrswiecik/dev/ives/coronary/sam2/workdir/arcade_classifier_data --resnet-checkpoint-dir /Users/piotrswiecik/dev/ives/coronary/sam2/workdir/artifacts/resnet
```

Single point inference test

```shell
uv run predict_random.py --weights-path /Users/piotrswiecik/dev/ives/coronary/sam2/workdir/artifacts/trained_models/large_v1/model.torch --dataset-root /Users/piotrswiecik/dev/ives/coronary/datasets/arcade

uv run predict_file.py --weights-path /Users/piotrswiecik/dev/ives/coronary/sam2/workdir/artifacts/trained_models/large_v1/model.torch --file /Users/piotrswiecik/dev/ives/coronary/datasets/wum_v2/1_I0468412.VIM.DCM.21.png
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

## Segment dictionary

1. RCA proximal
2. RCA mid
3. RCA distal
4. Posterior descending
5. Left main
6. LAD proximal
7. LAD mid
8. LAD apical
9. First diagonal
10. Second diagonal
11. Proximal circumflex
12. Intermediate/anterolateral
13. Distal circumflex
14. Left posterolateral
15. Posterior descending
16. Posterolateral from RCA

## ARCADE-to-Syntax mapping

```text
--- CATEGORY DEFINITIONS ---
ID: 1 | Name: 1
ID: 2 | Name: 2
ID: 3 | Name: 3
ID: 4 | Name: 4
ID: 5 | Name: 5
ID: 6 | Name: 6
ID: 7 | Name: 7
ID: 8 | Name: 8
ID: 9 | Name: 9
ID: 10 | Name: 9a
ID: 11 | Name: 10
ID: 12 | Name: 10a
ID: 13 | Name: 11
ID: 14 | Name: 12
ID: 15 | Name: 12a
ID: 16 | Name: 13
ID: 17 | Name: 14
ID: 18 | Name: 14a
ID: 19 | Name: 15
ID: 20 | Name: 16
ID: 21 | Name: 16a
ID: 22 | Name: 16b
ID: 23 | Name: 16c
ID: 24 | Name: 12b
ID: 25 | Name: 14b
ID: 26 | Name: stenosis
```