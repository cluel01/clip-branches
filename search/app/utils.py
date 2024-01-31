import os
import torch
import torch.nn.functional as F
import numpy as np
import time

def store_results(array,search_id,storage_path):
    filename = os.path.join(storage_path,search_id+".npy")
    np.save(filename,array)

    #delete old files
    current_time = time.time()
    for filename in os.listdir(storage_path):
        filepath = os.path.join(storage_path, filename)
        if os.path.isfile(filepath) and (current_time - os.path.getmtime(filepath)) > 600: #10 minutes
            os.remove(filepath)

def quantize_features(features,normalize=False,eps=1e-8):
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    
    if normalize:
        features = F.normalize(features, eps=eps, p=2, dim=-1) #L2 norm
    features = features * 127.5 + 128
    features = features.clamp(0, 255).byte()
    return features.cpu().numpy()
