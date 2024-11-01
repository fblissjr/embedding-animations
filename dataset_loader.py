import os
import pickle
import numpy as np
from tqdm import tqdm
from datasets import load_dataset as hf_load_dataset
from sklearn.preprocessing import StandardScaler

def load_mnist_data(dataset_name):
    """Load MNIST data from pickle files"""
    data_path = f'./data/{dataset_name}.pkl'
    if not os.path.exists(data_path):
        raise ValueError(f"MNIST dataset {dataset_name} not found. Please run data/mnist_np.py first.")
    
    print(f"Loading MNIST data from {data_path}...")
    with open(data_path, 'rb') as f:
        X, labels = pickle.load(f)
    return X, labels

def process_hf_dataset(dataset_name, split=None, max_samples=None, embed_fields=None, embed_model=None, embed_task=None, label_field=None):
    """Load and process a Hugging Face dataset for t-SNE visualization"""
    from embedding_utils import EmbeddingProcessor

    # Load the dataset using the HuggingFace datasets library
    print(f"\nLoading dataset {dataset_name}...")
    dataset = hf_load_dataset(dataset_name)
    
    # Get the appropriate split
    if split is None:
        split = list(dataset.keys())[0]
        print(f"No split specified. Using '{split}'. Available splits: {list(dataset.keys())}")
    elif split not in dataset:
        raise ValueError(f"Split '{split}' not found. Available splits: {list(dataset.keys())}")
    
    data = dataset[split]
    print(f"Using split '{split}' with {len(data)} examples")
    
    # If max_samples is specified, take a random sample
    if max_samples and max_samples < len(data):
        print(f"\nSampling {max_samples} examples from dataset...")
        indices = np.random.choice(len(data), max_samples, replace=False)
        data = data.select(indices)
    
    # Create labels based on the specified field or use simple numbering
    print("\nProcessing labels...")
    if label_field:
        if label_field not in data.features:
            raise ValueError(f"Label field '{label_field}' not found. Available fields: {list(data.features.keys())}")
        
        # Get unique values for the field
        values = data[label_field]
        
        # Handle different types of label fields
        if isinstance(values[0], (int, bool)):
            # For numeric/boolean fields, use as is
            labels = np.array(values)
        elif isinstance(values[0], float):
            # For continuous values, bin them into 10 categories
            print("Binning continuous values into 10 categories...")
            from sklearn.preprocessing import KBinsDiscretizer
            discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
            labels = discretizer.fit_transform(np.array(values).reshape(-1, 1)).ravel()
        else:
            # For string/categorical fields, map to integers
            print("Converting categorical labels to integers...")
            unique_values = sorted(set(values))
            label_map = {val: i for i, val in enumerate(unique_values)}
            labels = np.array([label_map[val] for val in tqdm(values, desc="Processing labels", unit="example")])
        
        print(f"Created labels from field '{label_field}' with {len(set(labels))} unique values")
    else:
        print("No label field specified. Using sequential numbering.")
        labels = np.arange(len(data))
    
    # Process features
    print("\nProcessing features...")
    if embed_model and embed_fields:
        processor = EmbeddingProcessor(embed_model, embed_task)
        X = processor.embed([dict(item) for item in data], embed_fields)
    else:
        raise ValueError("Embedding model and fields must be specified")
    
    # Normalize features
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, labels

def load_dataset(dataset_name, split=None, max_samples=None, embed_fields=None, embed_model=None, embed_task=None, label_field=None):
    """Universal dataset loader that handles both MNIST and Hugging Face datasets"""
    if dataset_name.startswith(('mnist', 'MNIST')):
        return load_mnist_data(dataset_name)
    else:
        return process_hf_dataset(
            dataset_name,
            split=split,
            max_samples=max_samples,
            embed_fields=embed_fields,
            embed_model=embed_model,
            embed_task=embed_task,
            label_field=label_field
        )