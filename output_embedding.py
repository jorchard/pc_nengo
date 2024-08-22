# output_embedding.py

import numpy as np
import pickle

def save_vectors(embedding_size=100, path="embedding_vectors.pickle"):
    v1 = np.random.random(embedding_size)
    v1 = v1/np.linalg.norm(v1)
    v2 = np.random.random(embedding_size)
    v2 = v2/np.linalg.norm(v1)

    d = {"Face Vector": v1,
         "Not Face Vector": v2,
         "size":embedding_size
        }
    with open(path, 'wb') as file:
        pickle.dump(d, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_vector(path="embedding_vectors.pickle"):
    with open(path, 'rb') as file:
        d = pickle.load(file)
    return d

if __name__=="__main__":
    save_vectors(100)