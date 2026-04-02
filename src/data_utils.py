"""
data_utils.py
Converts DeepScores V2 annotations to YOLO format for YOLOv8 training.
"""

import sys
sys.path.insert(0, '/Users/samyu/obb_anns')

from obb_anns import OBBAnns
from pathlib import Path
import shutil
import yaml
from tqdm import tqdm

# In-scope classes in a fixed order — index = YOLO class ID
IN_SCOPE = [
    'noteheadBlackOnLine', 'noteheadBlackInSpace',
    'noteheadHalfOnLine', 'noteheadHalfInSpace',
    'noteheadWholeOnLine', 'noteheadWholeInSpace',
    'ledgerLine', 'stem', 'beam',
    'flag8thDown', 'flag8thUp',
    'flag16thDown', 'flag16thUp',
    'restQuarter', 'restHalf', 'restWhole', 'rest8th',
    'clefG', 'clefF',
    'timeSig4', 'timeSig3', 'timeSig2', 'timeSigCommon', 'timeSigCutCommon',
    'accidentalSharp', 'accidentalFlat', 'accidentalNatural',
    'keySharp', 'keyFlat',
    'augmentationDot'
]

CLASS_TO_ID = {name: i for i, name in enumerate(IN_SCOPE)}


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert [x0, y0, x1, y1] absolute bbox to 
    YOLO format [cx, cy, w, h] normalized 0-1.
    """
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0 / img_width
    cy = (y0 + y1) / 2.0 / img_height
    w  = (x1 - x0) / img_width
    h  = (y1 - y0) / img_height
    # Clamp to [0, 1] to handle any edge annotations
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.0, min(1.0, w))
    h  = max(0.0, min(1.0, h))
    return cx, cy, w, h


def convert_split(ann_obj, img_src_dir, out_img_dir, out_label_dir, split_name):
    """
    Convert one split (train or test) to YOLO format.
    Copies images and writes label .txt files.
    """
    out_img_dir = Path(out_img_dir)
    out_label_dir = Path(out_label_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    img_src_dir = Path(img_src_dir)

    # Build lookups
    deepscores_cats = {
        cat_id: cat for cat_id, cat in ann_obj.cat_info.items()
        if cat['annotation_set'] == 'deepscores'
    }
    id_to_name = {str(cat_id): cat['name'] for cat_id, cat in deepscores_cats.items()}
    in_scope_ids = {str(cat_id) for cat_id, cat in deepscores_cats.items()
                    if cat['name'] in CLASS_TO_ID}
    img_lookup = {img['id']: img for img in ann_obj.img_info}

    # Group annotations by image
    print(f"Grouping annotations for {split_name} split...")
    img_annotations = {}
    for _, ann in tqdm(ann_obj.ann_info.iterrows(), total=len(ann_obj.ann_info)):
        cat_id = ann['cat_id'][0]
        if cat_id not in in_scope_ids:
            continue
        img_id = int(ann['img_id'])
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)

    # Write YOLO files
    print(f"Writing YOLO labels for {split_name} split...")
    skipped = 0
    for img_id, anns in tqdm(img_annotations.items()):
        img_info = img_lookup[img_id]
        img_width = img_info['width']
        img_height = img_info['height']
        filename = img_info['filename']

        # Copy image
        src = img_src_dir / filename
        dst = out_img_dir / filename
        if not dst.exists():
            shutil.copy2(src, dst)

        # Write label file
        label_file = out_label_dir / (Path(filename).stem + '.txt')
        with open(label_file, 'w') as f:
            for ann in anns:
                cat_id = ann['cat_id'][0]
                class_name = id_to_name[cat_id]
                class_id = CLASS_TO_ID[class_name]
                bbox = ann['a_bbox']
                x0, y0, x1, y1 = bbox
                if x1 <= x0 or y1 <= y0:
                    skipped += 1
                    continue
                cx, cy, w, h = convert_bbox_to_yolo(bbox, img_width, img_height)
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    print(f"  Done. Skipped {skipped} degenerate boxes.")
    return len(img_annotations)


def write_dataset_yaml(out_dir, train_img_dir, val_img_dir):
    """Write the dataset.yaml config file YOLOv8 needs."""
    yaml_content = {
        'path': str(Path(out_dir).resolve()),
        'train': str(Path(train_img_dir).resolve()),
        'val': str(Path(val_img_dir).resolve()),
        'nc': len(IN_SCOPE),
        'names': IN_SCOPE
    }
    yaml_path = Path(out_dir) / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print(f"dataset.yaml written to {yaml_path}")
    return yaml_path


if __name__ == '__main__':
    # Paths — adjust if your structure differs
    DATA_DIR      = Path('data/ds2_dense')
    IMG_DIR       = DATA_DIR / 'images'
    OUT_DIR       = Path('data/yolo_dataset')

    TRAIN_ANN     = DATA_DIR / 'deepscores_train.json'
    TEST_ANN      = DATA_DIR / 'deepscores_test.json'

    TRAIN_IMG_OUT = OUT_DIR / 'images/train'
    TRAIN_LBL_OUT = OUT_DIR / 'labels/train'
    VAL_IMG_OUT   = OUT_DIR / 'images/val'
    VAL_LBL_OUT   = OUT_DIR / 'labels/val'

    # Convert train split
    print("Loading train annotations...")
    train_anns = OBBAnns(str(TRAIN_ANN))
    train_anns.load_annotations()
    n_train = convert_split(train_anns, IMG_DIR, TRAIN_IMG_OUT, TRAIN_LBL_OUT, 'train')
    print(f"Train: {n_train} images converted\n")

    # Convert test split (used as val for YOLOv8)
    print("Loading test annotations...")
    test_anns = OBBAnns(str(TEST_ANN))
    test_anns.load_annotations()
    n_val = convert_split(test_anns, IMG_DIR, VAL_IMG_OUT, VAL_LBL_OUT, 'val')
    print(f"Val: {n_val} images converted\n")

    # Write dataset.yaml
    write_dataset_yaml(OUT_DIR, TRAIN_IMG_OUT, VAL_IMG_OUT)
    print("\nConversion complete.")
    print(f"YOLO dataset written to {OUT_DIR}")