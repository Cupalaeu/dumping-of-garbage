import os
import shutil
import json
import yaml
import numpy as np
import pandas as pd

def run_split(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42, base_dir=None):
    """
    Splits the manually curated dataset into stratified train, validation, and test sets.
    Only considers images that are still present in data/visual/.
    """
    if base_dir is None:
        # Resolve the root directory relative to this script (src/data/split.py)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        
    visual_dir = os.path.join(base_dir, "data", "visual")
    processed_dir = os.path.join(base_dir, "data", "processed")
    processed_images_dir = os.path.join(processed_dir, "images")
    processed_labels_dir = os.path.join(processed_dir, "labels")
    metadata_dir = os.path.join(base_dir, "data", "metadata")
    dataset_final_dir = os.path.join(base_dir, "data", "dataset_final")
    
    # 1. Verify requirements
    traceability_path = os.path.join(metadata_dir, "traceability.csv")
    if not os.path.exists(traceability_path):
        print(f"Error: Traceability file not found at {traceability_path}. Execute consolidation first.")
        return
        
    if not os.path.exists(visual_dir):
        print(f"Error: Visual directory not found at {visual_dir}.")
        return
        
    # 2. Get list of remaining visual files (user's selection)
    visual_files = os.listdir(visual_dir)
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG')
    visual_basenames = {
        os.path.splitext(f)[0] for f in visual_files if f.lower().endswith(image_extensions)
    }
    
    print(f"Total curated images in data/visual/: {len(visual_basenames)}")
    if len(visual_basenames) == 0:
        print("Error: No images found in data/visual/. Please curate your dataset first.")
        return
        
    # 3. Load and filter traceability
    df_trace = pd.read_csv(traceability_path)
    df_curated = df_trace[df_trace['new_name'].isin(visual_basenames)].copy()
    
    print(f"Traceability records matched: {len(df_curated)} / {len(df_trace)}")
    if len(df_curated) == 0:
        print("Error: None of the curated images matched traceability records.")
        return
        
    # Normalize ratios to sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio
    
    # 4. Perform stratified split
    # Stratified split ensures that the proportion of each source dataset is maintained in each split.
    df_curated['split'] = 'train'
    
    rng = np.random.RandomState(seed)
    
    split_dfs = []
    
    # Group by original dataset source
    for dataset_name, group in df_curated.groupby('original_dataset'):
        indices = group.index.tolist()
        rng.shuffle(indices)
        
        n_total = len(indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        # In case integer truncation leaves test split empty for tiny sets, distribute leftovers
        if len(test_idx) == 0 and n_total >= 3 and test_ratio > 0:
            test_idx = [indices[-1]]
            if len(val_idx) > 1:
                val_idx = val_idx[:-1]
            else:
                train_idx = train_idx[:-1]
                
        group.loc[train_idx, 'split'] = 'train'
        group.loc[val_idx, 'split'] = 'val'
        group.loc[test_idx, 'split'] = 'test'
        
        split_dfs.append(group)
        
    df_final = pd.concat(split_dfs)
    
    # 5. Re-create final dataset directories
    splits = ['train', 'val', 'test']
    for s in splits:
        s_img_dir = os.path.join(dataset_final_dir, s, "images")
        s_lbl_dir = os.path.join(dataset_final_dir, s, "labels")
        if os.path.exists(os.path.join(dataset_final_dir, s)):
            shutil.rmtree(os.path.join(dataset_final_dir, s))
        os.makedirs(s_img_dir, exist_ok=True)
        os.makedirs(s_lbl_dir, exist_ok=True)
        
    # 6. Copy files to the final dataset splits
    print("\nCopying files to splits...")
    copied_count = 0
    for idx, row in df_final.iterrows():
        new_basename = row['new_name']
        orig_name = row['original_name']
        split = row['split']
        
        ext = os.path.splitext(orig_name)[1].lower()
        
        # Sources
        src_img = os.path.join(processed_images_dir, f"{new_basename}{ext}")
        src_lbl = os.path.join(processed_labels_dir, f"{new_basename}.txt")
        
        # Destinations
        dest_img = os.path.join(dataset_final_dir, split, "images", f"{new_basename}{ext}")
        dest_lbl = os.path.join(dataset_final_dir, split, "labels", f"{new_basename}.txt")
        
        if os.path.exists(src_img) and os.path.exists(src_lbl):
            shutil.copy2(src_img, dest_img)
            shutil.copy2(src_lbl, dest_lbl)
            copied_count += 1
        else:
            print(f"Warning: Files missing for {new_basename} (img: {os.path.exists(src_img)}, lbl: {os.path.exists(src_lbl)})")
            
    print(f"Finished copying {copied_count} images and labels.")
    
    # 7. Save traceability_final.csv
    traceability_final_path = os.path.join(metadata_dir, "traceability_final.csv")
    df_final.to_csv(traceability_final_path, index=False, encoding='utf-8')
    print(f"Saved traceability_final CSV to {traceability_final_path}")
    
    # 8. Save unified dataset data.yaml
    unified_yaml = {
        "names": ["lixo"],
        "nc": 1,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images"
    }
    unified_yaml_path = os.path.join(dataset_final_dir, "data.yaml")
    with open(unified_yaml_path, 'w', encoding='utf-8') as yf:
        yaml.dump(unified_yaml, yf, default_flow_style=False)
    print(f"Saved unified data.yaml to {unified_yaml_path}")
    
    # 9. Print final split statistics
    print("\n" + "="*40)
    print("           SPLIT STATISTICS")
    print("="*40)
    print(f"Total dataset: {len(df_final)}")
    print(df_final['split'].value_counts())
    print("-"*40)
    print("Stratification check (distribution per dataset source):")
    pivot = df_final.pivot_table(index='original_dataset', columns='split', aggfunc='size', fill_value=0)
    total_series = pivot.sum(axis=1)
    for s in splits:
        if s in pivot.columns:
            pivot[f'{s}_%'] = (pivot[s] / total_series * 100).round(1)
    print(pivot)
    print("="*40)

if __name__ == "__main__":
    run_split()
