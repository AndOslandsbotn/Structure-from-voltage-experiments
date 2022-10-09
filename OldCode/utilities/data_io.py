from pathlib import Path
import numpy as np
import os
import json

def save_config(folder, config, bw, rhoG, rs):
    filepath = os.path.join('Results', folder)
    Path(filepath).mkdir(parents=True, exist_ok=True)

    config['bw'] = bw
    config['rhoG'] = rhoG
    config['rs'] = rs

    # Serializing json
    json_object = json.dumps(config, indent=4)

    # Writing to sample.json
    with open(os.path.join(filepath, "config.json"), "w") as outfile:
        outfile.write(json_object)
    return

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