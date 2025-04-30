import os
import random
import shutil

def count_images_in_directory(root_dir, extensions=('.jpg', '.jpeg', '.png')):
    count = 0
    for subdir, dirs, files in os.walk(root_dir):
        count += len([f for f in files if f.lower().endswith(extensions)])
    return count

def split_dataset(master_dir, output_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42, extensions=('.jpg', '.jpeg', '.png')):
    # Get a list of class names from the master directory
    classes = [d for d in os.listdir(master_dir) if os.path.isdir(os.path.join(master_dir, d))]
    print("Classes found:", classes)
    
    # Create output folder structure
    for split in ['Train', 'Valid', 'Test']:
        split_path = os.path.join(output_dir, split)
        os.makedirs(split_path, exist_ok=True)
        for cls in classes:
            os.makedirs(os.path.join(split_path, cls), exist_ok=True)
    
    # For each class, combine images from train, valid, test subfolders and re-split them
    for cls in classes:
        print(f"\nProcessing class: {cls}")
        source_class_dir = os.path.join(master_dir, cls)
        # Gather all image file paths from subfolders (assumes subfolders are named 'train', 'valid', 'test')
        all_files = []
        for sub_split in ['train', 'valid', 'test']:
            subfolder = os.path.join(source_class_dir, sub_split)
            if os.path.exists(subfolder):
                files = [os.path.join(subfolder, f) for f in os.listdir(subfolder) 
                         if f.lower().endswith(extensions)]
                all_files.extend(files)
        total = len(all_files)
        print(f"Total images in class '{cls}': {total}")
        
        if total == 0:
            print(f"No images found for class {cls}, skipping...")
            continue

        # Shuffle files
        random.seed(seed)
        random.shuffle(all_files)

        # Determine counts for each split
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        test_count = total - train_count - val_count

        print(f"Splitting: Train = {train_count}, Valid = {val_count}, Test = {test_count}")

        # Split file lists
        train_files = all_files[:train_count]
        val_files = all_files[train_count:train_count + val_count]
        test_files = all_files[train_count + val_count:]
        
        # Copy files to corresponding output directories
        for file in train_files:
            shutil.copy(file, os.path.join(output_dir, 'Train', cls, os.path.basename(file)))
        for file in val_files:
            shutil.copy(file, os.path.join(output_dir, 'Valid', cls, os.path.basename(file)))
        for file in test_files:
            shutil.copy(file, os.path.join(output_dir, 'Test', cls, os.path.basename(file)))
    
    # Print the distribution for each class
    for split in ['Train', 'Valid', 'Test']:
        print(f"\n{split} set distribution:")
        for cls in classes:
            count = count_images_in_directory(os.path.join(output_dir, split, cls), extensions)
            print(f"  {cls}: {count}")

if __name__ == "__main__":
    # Define master directory and output directory
    master_dir = r"D:/Research project/Datasets/master"   # Replace with your master directory path
    output_dir = r"D:/Research project/Datasets/Split70_15_15"  # New directory for the re-split dataset
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Split the dataset
    split_dataset(master_dir, output_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
