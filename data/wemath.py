from datasets import load_dataset, concatenate_datasets, Dataset
from datasets import Features, Value, Sequence, Image
from collections import Counter
import re
import gc
import os
import random
import numpy as np


def load_data(dataset_name, split_name):
    """Load a dataset with simple error handling."""
    try:
        dataset = load_dataset(dataset_name, split=split_name)
        return dataset
    except Exception as e:
        print(f"Failed to load {dataset_name}: {e}")
        return None


def check_image_structure(dataset, dataset_name):
    """Check how many images each entry contains in the dataset."""
    print(f"\n--- Image Structure Analysis for {dataset_name} ---")
    
    # Get a few sample entries to analyze
    sample_entries = dataset.select(range(min(5, len(dataset))))
    
    for i, entry in enumerate(sample_entries):
        print(f"\nSample {i+1}:")
        
        # Check image-related features
        image_features = [key for key in entry.keys() if 'image' in key.lower()]
        
        for feature in image_features:
            value = entry[feature]
            if hasattr(value, '__len__'):
                if isinstance(value, (list, tuple)):
                    print(f"  {feature}: {len(value)} images")
                    if len(value) > 0:
                        print(f"    First image type: {type(value[0])}")
                else:
                    print(f"  {feature}: 1 image (type: {type(value)})")
            else:
                print(f"  {feature}: 1 image (type: {type(value)})")
        
        # Also check if there are any other features that might contain images
        other_features = [key for key in entry.keys() if key not in image_features and key not in ['question', 'answer', 'options', 'choices']]
        if other_features:
            print(f"  Other features: {other_features}")


def merge_datasets_to_unified_format(datasets_dict, save_dir="merged_unified_dataset"):
    """
    Merge all datasets into a unified format with features: ['problem', 'images', 'answer']
    Process in batches to avoid memory issues and save incrementally to disk.
    """
    MAPPING = {"A": 0, "B": 1, "C": 2, "D": 3}
    
    print("\n=== Starting Dataset Merging Process ===")
    
    # Define unified features
    unified_features = Features({
        "problem": Value("string"),
        "images": Sequence(Image()),
        "answer": Value("string"),
    })
    
    # Create empty dataset to start with
    unified_dataset = Dataset.from_dict({
        "problem": [],
        "images": [],
        "answer": []
    }, features=unified_features)
    
    total_entries = 0
    
    # Process each dataset
    for dataset_name, dataset in datasets_dict.items():
        print(f"\nProcessing {dataset_name}...")
        
        if dataset_name == "lmms-lab/ai2d":
            entries = process_ai2d_dataset_batch(dataset, MAPPING)
        elif dataset_name == "Osilly/Vision-R1-rl":
            entries = process_vision_r1_dataset_batch(dataset)
        elif dataset_name == "HuggingFaceM4/A-OKVQA":
            entries = process_aokvqa_dataset_batch(dataset, MAPPING)
        elif dataset_name == "lmms-lab/textvqa":
            entries = process_textvqa_dataset_batch(dataset)
        else:
            print(f"Unknown dataset: {dataset_name}")
            continue
        
        # Convert entries to dataset format and concatenate
        print(f"  Converting {len(entries)} entries to dataset format...")
        
        # Process in smaller chunks to avoid memory issues
        chunk_size = 500
        for chunk_start in range(0, len(entries), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(entries))
            chunk_entries = entries[chunk_start:chunk_end]
            
            print(f"    Converting chunk {chunk_start//chunk_size + 1}: entries {chunk_start}-{chunk_end-1}")
            
            # Extract data from chunk
            problems = [entry["problem"] for entry in chunk_entries]
            images = [entry["images"] for entry in chunk_entries]
            answers = [entry["answer"] for entry in chunk_entries]
            
            # Create dataset from this chunk
            chunk_dataset = Dataset.from_dict({
                "problem": problems,
                "images": images,
                "answer": answers
            }, features=unified_features)
            
            # Concatenate with existing dataset
            unified_dataset = concatenate_datasets([unified_dataset, chunk_dataset])
            
            # Force garbage collection after each chunk
            gc.collect()
        
        total_entries += len(entries)
        print(f"  Added {len(entries)} entries from {dataset_name}")
        print(f"  Total entries so far: {total_entries}")
        
        # Force garbage collection to free memory
        gc.collect()
    
    print(f"\n=== Dataset Merging Complete ===")
    print(f"Total unified entries: {total_entries}")
    
    # Save to disk
    print(f"\n=== Saving to Disk ===")
    unified_dataset.save_to_disk(save_dir)
    print(f"Saved unified dataset to: {save_dir}")
    
    return unified_dataset


def process_ai2d_dataset_batch(dataset, MAPPING):
    """Process AI2D dataset to unified format in batches."""
    print(f"  Processing {len(dataset)} AI2D entries...")
    processed_entries = []
    batch_size = 500
    
    for start_idx in range(0, len(dataset), batch_size):
        end_idx = min(start_idx + batch_size, len(dataset))
        batch = dataset.select(range(start_idx, end_idx))
        
        print(f"    Processing batch {start_idx//batch_size + 1}: entries {start_idx}-{end_idx-1}")
        
        for entry in batch:
            # Get the question and options
            question = entry['question']
            options = entry['options']
            
            # Combine question and options
            problem_text = question
            if options and len(options) >= 4:
                problem_text += f"\nA) {options[0]}"
                problem_text += f"\nB) {options[1]}"
                problem_text += f"\nC) {options[2]}"
                problem_text += f"\nD) {options[3]}"
            
            # Get the answer
            answer = entry['answer']
            
            if isinstance(answer, str) and answer in MAPPING:
                # Already A, B, C, D format
                answer_text = answer
            elif isinstance(answer, int) and 0 <= answer < 4:
                # Convert numeric index (0, 1, 2, 3) to letter (A, B, C, D)
                answer_text = ["A", "B", "C", "D"][answer]
            elif isinstance(answer, str) and answer in ["0", "1", "2", "3"]:
                # String representation of numeric index
                answer_text = ["A", "B", "C", "D"][int(answer)]
            else:
                answer_text = str(answer)
            
            # Create unified entry
            unified_entry = {
                "images": [entry['image']],
                "problem": "<image>" + problem_text,
                "answer": answer_text
            }
            processed_entries.append(unified_entry)
    
    print(f"Processed {len(processed_entries)} AI2D entries")
    
    # Debug: Show a few examples of answer mapping
    if processed_entries:
        print("  AI2D Answer mapping examples:")
        for i in range(min(3, len(processed_entries))):
            entry = processed_entries[i]
            print(f"    Example {i+1}: Answer = '{entry['answer']}'")
    
    return processed_entries


def process_vision_r1_dataset_batch(dataset):
    """Process Vision-R1 dataset to unified format in batches."""
    print(f"  Processing {len(dataset)} Vision-R1 entries...")
    processed_entries = []
    batch_size = 1000
    
    for start_idx in range(0, len(dataset), batch_size):
        end_idx = min(start_idx + batch_size, len(dataset))
        batch = dataset.select(range(start_idx, end_idx))
        
        print(f"    Processing batch {start_idx//batch_size + 1}: entries {start_idx}-{end_idx-1}")
        
        for entry in batch:
            # Vision-R1 already has the desired format, check if <image> tag already exists
            problem_text = entry['problem']
            if not problem_text.startswith('<image>'):
                problem_text = "<image>" + problem_text
            
            # Handle the image properly - ensure it's a PIL Image
            image = entry['images']
            if isinstance(image, list):
                # Already a list, check if it contains PIL Images
                if len(image) > 0 and hasattr(image[0], 'convert'):
                    image_list = image  # Already in correct format
                else:
                    # List contains non-PIL objects, try to convert
                    try:
                        from PIL import Image
                        if hasattr(image[0], 'numpy'):  # numpy array
                            image_pil = Image.fromarray(image[0].numpy())
                        else:
                            image_pil = Image.fromarray(image[0])
                        image_list = [image_pil]
                    except Exception as e:
                        print(f"Warning: Could not convert image from list: {e}")
                        continue
            elif hasattr(image, 'convert'):  # Already a PIL Image
                image_list = [image]
            else:
                # Single non-PIL object, try to convert
                try:
                    from PIL import Image
                    if hasattr(image, 'numpy'):  # numpy array
                        image_pil = Image.fromarray(image.numpy())
                    else:
                        image_pil = Image.fromarray(image)
                    image_list = [image_pil]
                except Exception as e:
                    print(f"Warning: Could not convert image: {e}")
                    # Skip this entry if image conversion fails
                    continue
            
            # Create unified entry
            unified_entry = {
                "images": image_list,
                "problem": problem_text,
                "answer": entry['answer']
            }
            processed_entries.append(unified_entry)
    
    print(f"Processed {len(processed_entries)} Vision-R1 entries")
    return processed_entries


def process_aokvqa_dataset_batch(dataset, MAPPING):
    """Process A-OKVQA dataset to unified format in batches."""
    print(f"  Processing {len(dataset)} A-OKVQA entries...")
    processed_entries = []
    batch_size = 1000
    
    for start_idx in range(0, len(dataset), batch_size):
        end_idx = min(start_idx + batch_size, len(dataset))
        batch = dataset.select(range(start_idx, end_idx))
        
        print(f"Processing batch {start_idx//batch_size + 1}: entries {start_idx}-{end_idx-1}")
        
        for entry in batch:
            # Get the question and choices
            question = entry['question']
            choices = entry['choices']
            
            # Combine question and choices
            problem_text = question
            if choices and len(choices) >= 4:
                problem_text += f"\nA) {choices[0]}"
                problem_text += f"\nB) {choices[1]}"
                problem_text += f"\nC) {choices[2]}"
                problem_text += f"\nD) {choices[3]}"
            
            # Get the correct answer
            correct_idx = entry['correct_choice_idx']
            if isinstance(correct_idx, int) and 0 <= correct_idx < 4:
                # Convert numeric index (0, 1, 2, 3) to letter (A, B, C, D)
                answer_text = ["A", "B", "C", "D"][correct_idx]
            elif isinstance(correct_idx, str) and correct_idx in ["A", "B", "C", "D"]:
                # Already A, B, C, D format
                answer_text = correct_idx
            else:
                # Fallback to direct answers if available
                answer_text = str(entry.get('direct_answers', [''])[0]) if entry.get('direct_answers') else "Unknown"
            
            # Create unified entry
            unified_entry = {
                "images": [entry['image']],
                "problem": "<image>" + problem_text,
                "answer": answer_text
            }
            processed_entries.append(unified_entry)
    
    print(f"Processed {len(processed_entries)} A-OKVQA entries")
    
    # Debug: Show a few examples of answer mapping
    if processed_entries:
        print("  A-OKVQA Answer mapping examples:")
        for i in range(min(3, len(processed_entries))):
            entry = processed_entries[i]
            print(f"    Example {i+1}: Answer = '{entry['answer']}'")
    
    return processed_entries


def process_textvqa_dataset_batch(dataset):
    """Process TextVQA dataset to unified format in batches."""
    print(f"  Processing {len(dataset)} TextVQA entries...")
    processed_entries = []
    batch_size = 2000
    
    for start_idx in range(0, len(dataset), batch_size):
        end_idx = min(start_idx + batch_size, len(dataset))
        batch = dataset.select(range(start_idx, end_idx))
        
        print(f"Processing batch {start_idx//batch_size + 1}: entries {start_idx}-{end_idx-1}")
        
        for entry in batch:
            # Get the question
            question = entry['question']
            
            # Get the most common answer (majority vote)
            answers = entry['answers']
            if isinstance(answers, list) and answers:
                # Count occurrences of each answer
                answer_counts = Counter(answers)
                # Get the most common answer
                most_common_answer = answer_counts.most_common(1)[0][0]
            else:
                most_common_answer = str(answers) if answers else "Unknown"
            
            # Create unified entry
            unified_entry = {
                "images": [entry['image']],
                "problem": "<image>" + question,
                "answer": most_common_answer
            }
            processed_entries.append(unified_entry)
    
    print(f"Processed {len(processed_entries)} TextVQA entries")
    return processed_entries


def concatenate_data(dataset_list):
    """Concatenate multiple datasets."""
    return concatenate_datasets(dataset_list)


def save_data(dataset, file_path):
    """Save dataset to disk."""
    dataset.save_to_disk(file_path)


def balanced_sample_datasets(datasets_dict, train_samples=10000, val_samples=1000, random_seed=42):
    """
    Sample data from multiple datasets in a balanced way.
    
    Args:
        datasets_dict: Dictionary of datasets
        train_samples: Number of training samples to sample
        val_samples: Number of validation samples to sample
        random_seed: Random seed for reproducibility
    
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    print(f"\n=== Balanced Sampling ===")
    print(f"Target: {train_samples} training samples, {val_samples} validation samples")
    
    # Calculate samples per dataset for balanced distribution
    num_datasets = len(datasets_dict)
    train_samples_per_dataset = train_samples // num_datasets
    val_samples_per_dataset = val_samples // num_datasets
    
    # Handle remainder samples
    train_remainder = train_samples % num_datasets
    val_remainder = val_samples % num_datasets
    
    print(f"Samples per dataset: {train_samples_per_dataset} train, {val_samples_per_dataset} val")
    if train_remainder > 0:
        print(f"Remainder train samples: {train_remainder}")
    if val_remainder > 0:
        print(f"Remainder val samples: {val_remainder}")
    
    train_entries = []
    val_entries = []
    MAPPING = {"A": 0, "B": 1, "C": 2, "D": 3}
    # Process each dataset
    for i, (dataset_name, dataset) in enumerate(datasets_dict.items()):
        print(f"\nProcessing {dataset_name} for balanced sampling...")
        
        # Process dataset to unified format
        if dataset_name == "lmms-lab/ai2d":
            entries = process_ai2d_dataset_batch(dataset, MAPPING)
        elif dataset_name == "Osilly/Vision-R1-rl":
            entries = process_vision_r1_dataset_batch(dataset)
        elif dataset_name == "HuggingFaceM4/A-OKVQA":
            entries = process_aokvqa_dataset_batch(dataset, MAPPING)
        elif dataset_name == "lmms-lab/textvqa":
            entries = process_textvqa_dataset_batch(dataset)
        else:
            print(f"Unknown dataset: {dataset_name}")
            continue
        
        print(f"  Total entries from {dataset_name}: {len(entries)}")
        
        # Shuffle entries
        random.shuffle(entries)
        
        # Calculate samples for this dataset
        current_train_samples = train_samples_per_dataset
        current_val_samples = val_samples_per_dataset
        
        # Add remainder samples to first few datasets
        if i < train_remainder:
            current_train_samples += 1
        if i < val_remainder:
            current_val_samples += 1
        
        # Ensure we don't exceed available samples
        total_needed = current_train_samples + current_val_samples
        if len(entries) < total_needed:
            print(f"  Warning: {dataset_name} only has {len(entries)} samples, but {total_needed} are needed")
            print(f"  Using all available samples: {len(entries)}")
            current_train_samples = len(entries) // 2
            current_val_samples = len(entries) - current_train_samples
        
        # Split into train and val
        train_entries.extend(entries[:current_train_samples])
        val_entries.extend(entries[current_train_samples:current_train_samples + current_val_samples])
        
        print(f"  Sampled: {current_train_samples} train, {current_val_samples} val")
    
    print(f"\nFinal sampling results:")
    print(f"  Training samples: {len(train_entries)}")
    print(f"  Validation samples: {len(val_entries)}")
    
    # Convert to datasets
    train_dataset = create_dataset_from_entries(train_entries)
    val_dataset = create_dataset_from_entries(val_entries)
    
    return train_dataset, val_dataset


def create_dataset_from_entries(entries):
    """Create a dataset from a list of entries using chunking for better performance."""
    if not entries:
        return Dataset.from_dict({
            "problem": [],
            "images": [],
            "answer": []
        })
    
    # Define unified features
    unified_features = Features({
        "problem": Value("string"),
        "images": Sequence(Image()),
        "answer": Value("string"),
    })
    
    # Create empty dataset to start with
    dataset = Dataset.from_dict({
        "problem": [],
        "images": [],
        "answer": []
    }, features=unified_features)
    
    # Process in chunks to avoid memory issues
    chunk_size = 500
    print(f"  Converting {len(entries)} entries to dataset format in chunks of {chunk_size}...")
    
    for chunk_start in range(0, len(entries), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(entries))
        chunk_entries = entries[chunk_start:chunk_end]
        
        print(f"    Converting chunk {chunk_start//chunk_size + 1}: entries {chunk_start}-{chunk_end-1}")
        
        # Extract data from chunk
        problems = [entry["problem"] for entry in chunk_entries]
        images = [entry["images"] for entry in chunk_entries]
        answers = [entry["answer"] for entry in chunk_entries]
        
        # Create dataset from this chunk
        chunk_dataset = Dataset.from_dict({
            "problem": problems,
            "images": images,
            "answer": answers
        }, features=unified_features)
        
        # Concatenate with existing dataset
        dataset = concatenate_datasets([dataset, chunk_dataset])
        
        # Force garbage collection after each chunk
        gc.collect()
    
    return dataset


def merge_all_with_balanced_val(datasets_dict, val_samples=1000, random_seed=42):
    """
    Merge all datasets for training but set aside balanced validation samples.
    
    Args:
        datasets_dict: Dictionary of datasets
        val_samples: Number of validation samples to set aside (balanced across datasets)
        random_seed: Random seed for reproducibility
    
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    print(f"\n=== Merge All with Balanced Validation ===")
    print(f"Target: All data for training, {val_samples} balanced validation samples")
    
    # Calculate validation samples per dataset for balanced distribution
    num_datasets = len(datasets_dict)
    val_samples_per_dataset = val_samples // num_datasets
    val_remainder = val_samples % num_datasets
    
    print(f"Validation samples per dataset: {val_samples_per_dataset}")
    if val_remainder > 0:
        print(f"Remainder validation samples: {val_remainder}")
    
    train_entries = []
    val_entries = []
    MAPPING = {"A": 0, "B": 1, "C": 2, "D": 3}
    # Process each dataset
    for i, (dataset_name, dataset) in enumerate(datasets_dict.items()):
        print(f"\nProcessing {dataset_name}...")
        
        # Process dataset to unified format
        if dataset_name == "lmms-lab/ai2d":
            entries = process_ai2d_dataset_batch(dataset, MAPPING)
        elif dataset_name == "Osilly/Vision-R1-rl":
            entries = process_vision_r1_dataset_batch(dataset)
        elif dataset_name == "HuggingFaceM4/A-OKVQA":
            entries = process_aokvqa_dataset_batch(dataset, MAPPING)
        elif dataset_name == "lmms-lab/textvqa":
            entries = process_textvqa_dataset_batch(dataset)
        else:
            print(f"Unknown dataset: {dataset_name}")
            continue
        
        print(f"  Total entries from {dataset_name}: {len(entries)}")
        
        # Shuffle entries
        random.shuffle(entries)
        
        # Calculate validation samples for this dataset
        current_val_samples = val_samples_per_dataset
        
        # Add remainder samples to first few datasets
        if i < val_remainder:
            current_val_samples += 1
        
        # Ensure we don't exceed available samples
        if len(entries) < current_val_samples:
            print(f"  Warning: {dataset_name} only has {len(entries)} samples, but {current_val_samples} validation samples are needed")
            print(f"  Using all available samples for validation: {len(entries)}")
            current_val_samples = len(entries)
        
        # Split: validation samples first, then all remaining for training
        val_entries.extend(entries[:current_val_samples])
        train_entries.extend(entries[current_val_samples:])
        
        print(f"  Split: {len(entries) - current_val_samples} train, {current_val_samples} val")
    
    print(f"\nFinal results:")
    print(f"  Training samples: {len(train_entries)} (all remaining data)")
    print(f"  Validation samples: {len(val_entries)} (balanced across datasets)")
    
    # Convert to datasets
    train_dataset = create_dataset_from_entries(train_entries)
    val_dataset = create_dataset_from_entries(val_entries)
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    # Define datasets to load
    datasets_to_load = [
        ("lmms-lab/ai2d", "test"),
        ("Osilly/Vision-R1-rl", "train"),
        ("HuggingFaceM4/A-OKVQA", "train"),
        ("lmms-lab/textvqa", "train"),
        # ScienceQA removed due to compatibility issues
    ]
    
    # Load datasets
    loaded_datasets = {}
    
    for dataset_name, split_name in datasets_to_load:
        print(f"Loading {dataset_name} ({split_name})...")
        dataset = load_data(dataset_name, split_name)
        if dataset is not None:
            loaded_datasets[dataset_name] = dataset
            print(f"✓ Loaded {dataset_name}: {len(dataset)} samples")
        else:
            print(f"✗ Failed to load {dataset_name}")
    
    print(f"\nSuccessfully loaded {len(loaded_datasets)} datasets")
    
    # Display dataset information and check image structure
    # for name, dataset in loaded_datasets.items():
    #     print(f"\n{name}:")
    #     print(f"  Samples: {len(dataset)}")
    #     print(f"  Features: {list(dataset.features.keys())}")
        
        # Check image structure for each dataset
        # check_image_structure(dataset, name)
    
    # Choose your sampling strategy:
    # Option 1: Balanced sampling (10,000 train + 1,000 val, balanced across datasets)
    # Option 2: Use all data for training, set aside 1,000 balanced validation samples
    
    save_dir = "/apdcephfs_gy2/share_302625456/user/rrrliu/datasets/mllm_rl_small"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Option 1: Balanced sampling (uncomment to use)
    train_dataset, val_dataset = balanced_sample_datasets(
        loaded_datasets, 
        train_samples=10000, 
        val_samples=1000, 
        random_seed=42
    )
    
    # Option 2: Use all data for training, balanced validation (currently active)
    # train_dataset, val_dataset = merge_all_with_balanced_val(
    #     loaded_datasets, 
    #     val_samples=1000, 
    #     random_seed=42
    # )
    
    # Save train and validation datasets separately
    train_save_path = os.path.join(save_dir, "train")
    val_save_path = os.path.join(save_dir, "val")
    
    print(f"\n=== Saving Balanced Datasets ===")
    print(f"Saving training dataset ({len(train_dataset)} samples) to: {train_save_path}")
    train_dataset.save_to_disk(train_save_path)
    
    print(f"Saving validation dataset ({len(val_dataset)} samples) to: {val_save_path}")
    val_dataset.save_to_disk(val_save_path)
    
    # Show a few samples from both datasets
    print("\n=== Sample from Training Dataset ===")
    for i in range(min(3, len(train_dataset))):
        ex = train_dataset[i]
        print(f"[{i}] problem: {ex['problem'][:100]}... | answer: {ex['answer']} | num_images: {len(ex['images'])}")
    
    print("\n=== Sample from Validation Dataset ===")
    for i in range(min(3, len(val_dataset))):
        ex = val_dataset[i]
        print(f"[{i}] problem: {ex['problem'][:100]}... | answer: {ex['answer']} | num_images: {len(ex['images'])}")
    
    # Show dataset statistics
    print(f"\n=== Dataset Statistics ===")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Total samples: {len(train_dataset) + len(val_dataset)}")
    print(f"Datasets used: {list(loaded_datasets.keys())}")
    print(f"Save directory: {save_dir}")