from pathlib import Path
import numpy as np
import os

def load_data(directory, filename):
    print("Load from file")
    path = os.path.join(directory, filename)
    try:
        npzfile = np.load(path + '.npz')
        data = npzfile[npzfile.files[0]]
        return data
    except OSError as err:
        print(f'{err} No file found with this name')

def save_data(data, directory, filename):
    print("Save to file")
    Path(os.path.join(directory)).mkdir(parents=True, exist_ok=True)
    path = os.path.join(directory, filename)
    np.savez_compressed(path + '.npz', p=data)
    return