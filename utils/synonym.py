import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from utils.embeddings_utils import get_embeddings


def is_synonym(new_term, model_type, model, tokenizer=None, max_seq_length=None, embeddings_synonyms=None, embeddings_ontology=None):
    """
    Compare a new_term with the list of synonyms and ontology terms stored using cosine similarity.

    Args:
        new_term: A single term to be compared.
        model_type: The type of model to use ('bert', 'fasttext', 'word2vec').
        model: The model to use for generating embeddings (DistilBERT, FastText, or Word2Vec model).
        tokenizer: The tokenizer associated with the BERT model (optional, required for 'bert' model_type).
        max_seq_length: The maximum length for tokenization (optional, required for 'bert' model_type).
        embeddings_synonyms: Dictionary of synonym embeddings.
        embeddings_ontology: Dictionary of ontology embeddings.

    Returns:
        A string indicating if the term is a synonym to any word in the list and the score, or if no synonym was found.
    """
    new_term_embds = get_embeddings([new_term], model_type, model, tokenizer, max_seq_length)
    similarity_scores = {}
    overall_embeddings = {**embeddings_synonyms, **embeddings_ontology}

    for term, embedding in overall_embeddings.items():
        if model_type == 'bert' or model_type == 'fasttext':
            similarity = cosine_similarity(new_term_embds[new_term].reshape(1, -1), embedding.reshape(1, -1))[0][0]
        elif model_type == 'word2vec':
            for new_term_token, new_term_embedding in new_term_embds.items():
                similarity = cosine_similarity(new_term_embedding.reshape(1, -1), embedding.reshape(1, -1))[0][0]
                similarity_scores[term] = similarity
                continue  # Continue to avoid overwriting similarity_scores

        similarity_scores[term] = similarity

    sorted_similarity_scores = dict(sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True))

    if sorted_similarity_scores and list(sorted_similarity_scores.values())[0] > 0.7:
        top_synonym = list(sorted_similarity_scores.keys())[0]
        return f"'{new_term}' is synonym to '{top_synonym}' with a similarity score of {list(sorted_similarity_scores.values())[0]:.2f}"
    else:
        return f"No close synonyms found in the list for the term: {new_term}"