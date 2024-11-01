from typing import List, Optional, Union, Dict
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

class EmbeddingProcessor:
    """Handles text embedding using various models with special handling for task-specific models."""
    
    # Known task-specific embedding models and their supported tasks
    TASK_SUPPORTED_MODELS = {
        'jinaai/jina-embeddings-v3': {
            'tasks': [
                'retrieval.query',
                'retrieval.passage',
                'separation',
                'classification',
                'text-matching'
            ],
            'default_task': 'text-matching'
        }
    }

    def __init__(self, model_name: str, task: Optional[str] = None):
        """Initialize the embedding processor.
        
        Args:
            model_name: Name of the embedding model (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
            task: Specific task for models that support different embedding tasks
        """
        print(f"Loading embedding model {model_name}...")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        
        # Handle task-specific models
        self.task = None
        if model_name in self.TASK_SUPPORTED_MODELS:
            model_info = self.TASK_SUPPORTED_MODELS[model_name]
            if task and task not in model_info['tasks']:
                raise ValueError(
                    f"Task '{task}' not supported for model '{model_name}'. "
                    f"Supported tasks: {model_info['tasks']}"
                )
            self.task = task or model_info['default_task']
            print(f"Using task: {self.task}")
    
    def _prepare_texts(self, texts: List[Dict[str, str]], fields: List[str]) -> List[str]:
        """Prepare texts for embedding by concatenating specified fields."""
        print("Processing text fields...")
        processed_texts = []
        for item in tqdm(texts, desc="Preparing texts", unit="example"):
            field_texts = []
            for field in fields:
                if field in item:
                    # Handle nested fields (e.g., "messages.content")
                    if isinstance(item[field], (list, dict)):
                        if isinstance(item[field], list):
                            # Handle list of messages (e.g., chat conversations)
                            field_text = " | ".join(str(msg) for msg in item[field])
                        else:
                            # Handle nested dictionary
                            field_text = str(item[field])
                    else:
                        field_text = str(item[field])
                    field_texts.append(field_text)
            processed_texts.append(" | ".join(field_texts))
        return processed_texts

    def embed(self, texts: List[Dict[str, str]], fields: List[str]) -> np.ndarray:
        """Embed the texts using the specified model and fields.
        
        Args:
            texts: List of dictionaries containing the text data
            fields: List of fields to use for embedding
            
        Returns:
            numpy.ndarray: Array of embeddings
        """
        processed_texts = self._prepare_texts(texts, fields)
        
        print("Generating embeddings...")
        # Use model.encode with show_progress_bar=True for built-in progress bar
        if self.task:
            embeddings = self.model.encode(
                processed_texts,
                task=self.task,
                prompt_name=self.task,
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=32
            )
        else:
            embeddings = self.model.encode(
                processed_texts,
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=32
            )
            
        return embeddings