#!/usr/bin/env python3
"""
Script to upload the unified dataset to HuggingFace Hub
"""

from datasets import load_from_disk
import os

def upload_dataset_to_hub():
    # Dataset path
    dataset_path = "/apdcephfs_gy2/share_302625456/user/rrrliu/datasets/mllm_rl_small"
    repo_id = "lr10260/mllm_rl_small"
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please run data_collect.py first to create the dataset")
        return
    
    # Load the dataset splits individually and combine into DatasetDict
    try:
        from datasets import DatasetDict
        
        # Load each split individually
        train_path = os.path.join(dataset_path, "train")
        val_path = os.path.join(dataset_path, "val")

        if os.path.exists(train_path):
            print(f"Loading train split from: {train_path}")
            train_dataset = load_from_disk(train_path)
            print(f"  Train: {len(train_dataset)} samples")
        
        if os.path.exists(val_path):
            print(f"Loading validation split from: {val_path}")
            val_dataset = load_from_disk(val_path)
            print(f"  Validation: {len(val_dataset)} samples")
        
        # Combine into DatasetDict if both splits exist
        if train_dataset and val_dataset:
            dataset = DatasetDict({
                "train": train_dataset,
                "val": val_dataset
            })
        elif train_dataset and not val_dataset:
            dataset = DatasetDict({
                "train": train_dataset
            })
        elif val_dataset and not train_dataset:
            dataset = DatasetDict({
                "val": val_dataset
            })
        else:
            return
        
        print(f"Created DatasetDict with splits: {list(dataset.keys())}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Add dataset metadata to each split
    print("\nAdding dataset metadata...")
    for split_name, split_dataset in dataset.items():
        split_dataset.info.description = "Vision-language dataset for RL training"
        split_dataset.info.license = "Apache 2.0"
        split_dataset.info.homepage = f"https://huggingface.co/datasets/{repo_id}"
        print(f"  Added metadata to {split_name} split")
    
    # Push to hub
    print(f"\nPushing dataset to: https://huggingface.co/datasets/{repo_id}")
    try:
        dataset.push_to_hub(
            repo_id=repo_id,
            private=True,
            commit_message="Add unified vision-language dataset"
        )
        print(f"✅ Successfully uploaded dataset!")
        print(f"Dataset available at: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"❌ Error uploading dataset: {e}")
        print("Make sure you have the correct permissions and are logged in")

if __name__ == "__main__":
    upload_dataset_to_hub() 