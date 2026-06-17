import os
import shutil
import json
import yaml
import cv2
import pandas as pd
import glob

def parse_readme(readme_path, folder_name):
    """
    Parses README.dataset.txt to extract dataset name, Roboflow URL, and license.
    """
    info = {
        "dataset_name": folder_name,
        "roboflow_url": "Unknown",
        "license": "Unknown",
        "extra_info": ""
    }
    
    if not os.path.exists(readme_path):
        return info
        
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
        if lines:
            # First line is usually '# Name > Date'
            if lines[0].startswith('#'):
                info["dataset_name"] = lines[0].lstrip('#').strip()
            
            # Look for URL and License in the rest of the lines
            for line in lines[1:]:
                if line.startswith('https://'):
                    info["roboflow_url"] = line
                elif 'license' in line.lower() or 'licença' in line.lower():
                    info["license"] = line.split(':')[-1].strip()
                else:
                    if info["extra_info"]:
                        info["extra_info"] += " | " + line
                    else:
                        info["extra_info"] = line
    except Exception as e:
        print(f"Warning: Failed to parse README at {readme_path}: {e}")
        
    return info

def draw_bounding_boxes(image_path, label_path, class_names, output_path):
    """
    Draws bounding boxes and class names on the image and saves it.
    Supports both standard YOLO bounding boxes (5 values) and YOLO segmentation polygons (>5 values).
    """
    img = cv2.imread(image_path)
    if img is None:
        return False
        
    h, w, _ = img.shape
    
    if os.path.exists(label_path):
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                
                if len(parts) == 5:
                    # Standard YOLO bbox: class_id x_center y_center width height
                    x_c, y_c, box_w, box_h = map(float, parts[1:5])
                    x1 = int((x_c - box_w / 2) * w)
                    y1 = int((y_c - box_h / 2) * h)
                    x2 = int((x_c + box_w / 2) * w)
                    y2 = int((y_c + box_h / 2) * h)
                else:
                    # YOLO Polygon segment: class_id x1 y1 x2 y2 ... xN yN
                    coords = list(map(float, parts[1:]))
                    xs = coords[0::2]
                    ys = coords[1::2]
                    
                    if not xs or not ys:
                        continue
                        
                    min_x = min(xs)
                    max_x = max(xs)
                    min_y = min(ys)
                    max_y = max(ys)
                    
                    x1 = int(min_x * w)
                    y1 = int(min_y * h)
                    x2 = int(max_x * w)
                    y2 = int(max_y * h)
                    
                    # Draw polygon lines
                    poly_points = []
                    for x, y in zip(xs, ys):
                        poly_points.append([int(x * w), int(y * h)])
                    import numpy as np
                    pts = np.array(poly_points, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], isClosed=True, color=(128, 128, 255), thickness=1)
                
                # Clamp coordinates to image boundaries
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))
                
                # Vibrant color (cyan) for bounding boxes
                color = (255, 191, 0)  # BGR format: vibrant blue-cyan
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Look up class name from original dataset's names
                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                label_text = f"{class_name}"
                
                # Professional typography scaling
                font_scale = max(0.4, min(w, h) / 800.0)
                thickness = max(1, int(min(w, h) / 600))
                
                # Drawing label background text container
                (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
                cv2.putText(img, label_text, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        except Exception as e:
            print(f"Warning: Failed to annotate {image_path} with labels from {label_path}: {e}")
            
    cv2.imwrite(output_path, img)
    return True

def clean_directory(directory_path):
    """
    Safely deletes all contents of a directory without deleting the directory itself.
    This prevents PermissionError (Access Denied) in syncing environments like OneDrive.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        return
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Warning: Could not delete {item_path}. It might be locked by OneDrive. Error: {e}")

def run_consolidation(base_dir=None):
    if base_dir is None:
        # Resolve the root directory relative to this script (src/data/consolidate.py)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")
    processed_images_dir = os.path.join(processed_dir, "images")
    processed_labels_dir = os.path.join(processed_dir, "labels")
    visual_dir = os.path.join(base_dir, "data", "visual")
    metadata_dir = os.path.join(base_dir, "data", "metadata")
    
    # Clean output directories to ensure a fresh run without leftover files from previous runs
    for d in [processed_images_dir, processed_labels_dir, visual_dir]:
        clean_directory(d)
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Find all original dataset folders in data/raw
    if not os.path.exists(raw_dir):
        print(f"Error: Raw directory does not exist at {raw_dir}")
        return
        
    dataset_folders = sorted([f for f in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, f))])
    print(f"Found {len(dataset_folders)} datasets to consolidate:")
    for f in dataset_folders:
        print(f" - {f}")
        
    mapping = {}
    traceability_records = []
    
    # Step 1: Parse datasets and generate mapping.json
    for idx, folder in enumerate(dataset_folders, start=1):
        alias = f"dt{idx}"
        folder_path = os.path.join(raw_dir, folder)
        
        # Parse README
        readme_path = os.path.join(folder_path, "README.dataset.txt")
        readme_info = parse_readme(readme_path, folder)
        
        # Parse data.yaml to get original classes
        yaml_path = os.path.join(folder_path, "data.yaml")
        classes = []
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r', encoding='utf-8') as yf:
                    data_yaml = yaml.safe_load(yf)
                    if data_yaml and "names" in data_yaml:
                        classes = data_yaml["names"]
            except Exception as e:
                print(f"Warning: Failed to parse {yaml_path}: {e}")
                
        mapping[alias] = {
            "alias": alias,
            "original_folder": folder,
            "dataset_name": readme_info["dataset_name"],
            "roboflow_url": readme_info["roboflow_url"],
            "license": readme_info["license"],
            "classes": classes
        }
        
    # Save mapping.json
    mapping_path = os.path.join(metadata_dir, "mapping.json")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"\nSaved mapping JSON to {mapping_path}")
    
    # Step 2: Iterate over each dataset to process and rename images/labels
    for alias, info in mapping.items():
        folder = info["original_folder"]
        folder_path = os.path.join(raw_dir, folder)
        classes = info["classes"]
        
        print(f"\nProcessing {alias} ({folder})...")
        
        # Keep track of sequential number for this specific dataset
        seq_num = 1
        
        # Roboflow datasets usually contain split directories: train, valid, test
        splits = ["train", "valid", "test"]
        for split in splits:
            split_path = os.path.join(folder_path, split)
            if not os.path.exists(split_path):
                continue
                
            images_src_dir = os.path.join(split_path, "images")
            labels_src_dir = os.path.join(split_path, "labels")
            
            if not os.path.exists(images_src_dir):
                # Sometimes split folder directly contains images (unlikely for YOLO, but good to handle)
                continue
                
            # Scan images in the images folder
            image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG", "*.JPEG"]
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(glob.glob(os.path.join(images_src_dir, ext)))
                
            # Remove duplicates (case-insensitive filesystem gotcha on Windows)
            image_paths = list(set(os.path.abspath(p) for p in image_paths))
            
            # Sort images to maintain a consistent sequence
            image_paths.sort()
            
            for img_path in image_paths:
                orig_filename = os.path.basename(img_path)
                ext = os.path.splitext(orig_filename)[1].lower()
                
                # New standardized names
                new_basename = f"lixo-{alias}-{seq_num:05d}"
                new_image_name = f"{new_basename}{ext}"
                new_label_name = f"{new_basename}.txt"
                
                dest_image_path = os.path.join(processed_images_dir, new_image_name)
                dest_label_path = os.path.join(processed_labels_dir, new_label_name)
                dest_visual_path = os.path.join(visual_dir, new_image_name)
                
                # Check for corresponding label file
                orig_label_basename = os.path.splitext(orig_filename)[0] + ".txt"
                orig_label_path = os.path.join(labels_src_dir, orig_label_basename)
                
                # Check if the label file is missing or empty
                is_empty = True
                if os.path.exists(orig_label_path) and os.path.getsize(orig_label_path) > 0:
                    try:
                        with open(orig_label_path, 'r', encoding='utf-8') as lf_in:
                            content = lf_in.read().strip()
                        if len(content) > 0:
                            is_empty = False
                    except Exception as e:
                        print(f"Warning checking label file {orig_label_path}: {e}")
                
                if is_empty:
                    # Skip image and label if label file is missing or empty
                    continue
                
                # 1. Copy image file
                shutil.copy2(img_path, dest_image_path)
                
                # 2. Process and copy label file (Unifying class IDs to 0)
                try:
                    with open(orig_label_path, 'r', encoding='utf-8') as lf_in:
                        lines = lf_in.readlines()
                        
                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # Set class ID to 0, preserve bbox coords
                            parts[0] = "0"
                            new_lines.append(" ".join(parts) + "\n")
                            
                    with open(dest_label_path, 'w', encoding='utf-8') as lf_out:
                        lf_out.writelines(new_lines)
                except Exception as e:
                    print(f"Error processing label file {orig_label_path}: {e}")
                    # If error, copy empty file
                    with open(dest_label_path, 'w', encoding='utf-8') as lf_out:
                        pass
                
                # 3. Draw bounding boxes on copy for manual QA
                # We use the original label path to draw the boxes with original class names
                draw_bounding_boxes(dest_image_path, orig_label_path, classes, dest_visual_path)
                
                # 4. Add to traceability records
                traceability_records.append({
                    "new_name": new_basename,
                    "original_dataset": folder,
                    "original_split": split,
                    "original_name": orig_filename
                })
                
                seq_num += 1
                
        print(f" - Processed {seq_num - 1} images/labels for {alias}")
        
    # Save traceability.csv
    traceability_df = pd.DataFrame(traceability_records)
    traceability_csv_path = os.path.join(metadata_dir, "traceability.csv")
    traceability_df.to_csv(traceability_csv_path, index=False, encoding='utf-8')
    print(f"\nSaved traceability CSV to {traceability_csv_path}")
    
    # Save a unified data.yaml in data/processed/
    unified_yaml = {
        "names": ["lixo"],
        "nc": 1,
        "train": "../processed/images",
        "val": "../processed/images"
    }
    unified_yaml_path = os.path.join(processed_dir, "data.yaml")
    with open(unified_yaml_path, 'w', encoding='utf-8') as yf:
        yaml.dump(unified_yaml, yf, default_flow_style=False)
    print(f"Saved unified data.yaml to {unified_yaml_path}")
    
    print("\nDataset consolidation completed successfully!")

if __name__ == "__main__":
    run_consolidation()
