"""This module creates augmented dataset for classifier training."""

import os
import cv2
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# --- CONFIG ---
TRAIN_JSON = "/Users/piotrswiecik/dev/ives/coronary/datasets/arcade/syntax/train/annotations/train.json"
TRAIN_IMAGES_DIR = "/Users/piotrswiecik/dev/ives/coronary/datasets/arcade/syntax/train/images"
OUTPUT_DIR = "workdir/arcade_classifier_data"

def generate_classifier_dataset(json_path, image_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Loading JSON from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    img_to_anns = defaultdict(list)
    for ann in data['annotations']:
        img_to_anns[ann['image_id']].append(ann)
        
    img_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    
    print(f"Processing {len(img_to_anns)} images to extract vessel crops...")
    
    count_saved = 0
    count_skipped = 0
    
    for img_id, anns in tqdm(img_to_anns.items()):
        if img_id not in img_id_to_filename: continue
        
        filename = img_id_to_filename[img_id]
        img_path = os.path.join(image_dir, filename)
        
        image = cv2.imread(img_path)
        if image is None: continue
        
        h, w = image.shape[:2]
        
        for ann in anns:
            cat_id = ann['category_id']
            
            class_dir = os.path.join(output_dir, str(cat_id))
            os.makedirs(class_dir, exist_ok=True)
            
            mask = np.zeros((h, w), dtype=np.uint8)
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [poly], color=1)
            
            masked_crop = np.zeros_like(image)
            masked_crop[mask == 1] = image[mask == 1]
            
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) == 0: 
                count_skipped += 1
                continue
            
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()
            
            pad = 10
            y_min = max(0, y_min - pad)
            y_max = min(h, y_max + pad)
            x_min = max(0, x_min - pad)
            x_max = min(w, x_max + pad)
            
            final_crop = masked_crop[y_min:y_max, x_min:x_max]
            
            if final_crop.shape[0] < 10 or final_crop.shape[1] < 10:
                count_skipped += 1
                continue
                
            save_name = f"{img_id}_{ann['id']}.png"
            cv2.imwrite(os.path.join(class_dir, save_name), final_crop)
            count_saved += 1
            
    print(f"\nDone. Saved {count_saved} crops. Skipped {count_skipped} (empty/tiny).")
    print(f"Data stored in: {output_dir}")


if __name__ == "__main__":  
    generate_classifier_dataset(TRAIN_JSON, TRAIN_IMAGES_DIR, OUTPUT_DIR)