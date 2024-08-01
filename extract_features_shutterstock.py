import torch
from torch.utils.data import DataLoader
from src.dataset import CLIP_Dataset
from src.models import CLIPModelWrapper
import numpy as np
from tqdm import tqdm
torch.multiprocessing.set_sharing_strategy('file_system')

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

seed = 3
torch.manual_seed(seed)
np.random.seed(seed)

bs = 512
embed_dim = 32
hidden_dim = 8192
device = "cuda:" if torch.cuda.is_available() else "cpu"
model = CLIPModelWrapper("ViT-B/32", device=device,freeze_backbone=True,
                         embed_dim=embed_dim,hidden_dim=hidden_dim,distance="euclidean",
                         )


model.load_state_dict(torch.load("assets/search/data/clip_weights.pth"))


dataset = CLIP_Dataset(root="data/datasets/shutterstock/images",labels_path="data/datasets/shutterstock_dataset.csv",
                       transform=model.preprocess)
loader = DataLoader(dataset, batch_size=bs, shuffle=False,num_workers=7)


model.eval()
features = np.zeros((len(dataset),embed_dim),dtype="float32")
with torch.no_grad():
    N = 0
    for i, data in tqdm(enumerate(loader, 0),total=len(loader)):
        img,_= data
        img = img.to(device)
        n = len(img)
        embedding = model.encode_image(img.half())
        features[N:N+n] = embedding.cpu().numpy()
        N += n
        # embeddings.append(embedding.cpu().numpy())
# features = np.vstack(embeddings)
np.save("assets/search/shutterstock_clip_features_d512.npy",features)