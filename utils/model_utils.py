
import pickle

def save_embeddings_to_pickle(embeddings, filename):
    """
    Saves embeddings to a pickle file.

    Args:
        embeddings: A dictionary mapping terms to their corresponding embedding vectors.
        filename: The name of the pickle file to save the embeddings.
    """
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)
        

def load_pickle_model(file_path):
    """
    Load a pickle model from the specified file path.

    Args:
        file_path (str): The path to the pickle file.

    Returns:
        object: The loaded model object.
    """
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model