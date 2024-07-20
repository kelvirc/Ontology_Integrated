import numpy as np
import torch
import re
from gensim.models import KeyedVectors, Word2Vec

def tokenize_word2vec(text):
    """
    Tokenizes the input text.

    Args:
        text: A string to be tokenized.

    Returns:
        A list of words.
    """
    # Basic preprocessing and tokenization
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    tokens = text.split()
    return tokens

def get_embeddings(terms, model_type, model, tokenizer=None, max_seq_length=None):
    """
    Generates embeddings for a list of terms using the specified model type.

    Args:
        terms: A list of strings or lists of strings (representing phrases).
        model_type: The type of model to use ('bert', 'fasttext', or 'word2vec').
        model: The model to use for generating embeddings (DistilBERT, FastText, or Word2Vec model).
        tokenizer: The tokenizer associated with the BERT model (optional, required for 'bert' model_type).
        max_seq_length: The maximum length for tokenization (optional, required for 'bert' model_type).

    Returns:
        A dictionary mapping each term to its corresponding embedding vector.
    """
    embeddings = {}
    
    if model_type == 'bert':
        if tokenizer is None or max_seq_length is None:
            raise ValueError("Tokenizer and max_seq_length must be provided for BERT model_type.")
        
        for term in terms:
            if isinstance(term, list):
                term = " ".join(term)

            inputs = tokenizer(
                term,
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                max_length=max_seq_length,
                return_tensors='pt'
            )

            with torch.no_grad():
                inputs = {key: val.to(model.device) for key, val in inputs.items()}
                outputs = model(**inputs)

            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings[term] = embedding[0]

    elif model_type == 'fasttext':
        for term in terms:
            words = tokenize_word2vec(term)
            word_vectors = [model[word] for word in words if word in model]
            if word_vectors:
                embedding = np.mean(word_vectors, axis=0)
                embeddings[term] = embedding
            else:
                embeddings[term] = np.zeros(model.vector_size)  # Correct attribute for FastText

    elif model_type == 'word2vec':
        for term in terms:
            tokens = term.split()
            for token in tokens:
                if token in model.wv:
                    embeddings[token] = model.wv[token]

    else:
        raise ValueError("Invalid model_type. Choose 'bert', 'fasttext', or 'word2vec'.")

    return embeddings