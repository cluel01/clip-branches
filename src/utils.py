import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

def quantize_features_uniform(features,normalize=False,eps=1e-8):
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    
    if normalize:
        features = F.normalize(features, eps=eps, p=2, dim=-1) #L2 norm
    features = features * 127.5 + 128
    features = features.clamp(0, 255).byte()
    return features.cpu().numpy()

def quantize_features_minmax(features):
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)

    min_vals = features.min(dim=0)[0]
    max_vals = features.max(dim=0)[0]

    # Perform min-max normalization
    features = (features - min_vals) / (max_vals - min_vals)

    features = features * 255
    features = features.clamp(0, 255).byte()
    return features.cpu().numpy()

def extract_image_features(model,loader,device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader, 0),total=len(loader)):
            img,_= data
            img = img.to(device)
        
            embedding = model.encode_image(img.half())
            embeddings.append(embedding.cpu().numpy())
    features = np.vstack(embeddings)
    return features