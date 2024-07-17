import os
import torch
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import clip
from PIL import Image

# Download and load the CIFAR10 dataset
dataset = CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())


# Iterate over the dataset and save each image in its corresponding class directory
for i, (image, label) in enumerate(dataset):
    # Define the directory name based on the class label
    lab = dataset.classes[label]
    label_dir = os.path.join('data/cifar10/', lab)
    os.makedirs(label_dir, exist_ok=True)

    # Convert the tensor image back to PIL image
    pil_image = transforms.ToPILImage()(image)

    # Define the image file path
    image_path = os.path.join(label_dir, f'image_test_{i}.png')

    # Save the image
    pil_image.save(image_path)
    
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

    
# Download and load the training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=preprocess)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=False, num_workers=2)

# Download and load the validation data
validationset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=preprocess)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=128,
                                         shuffle=False, num_workers=2)

model.eval()
embeddings = []
with torch.no_grad():
    for i, data in tqdm(enumerate(trainloader, 0),total=len(trainloader)):
        img= data[0]
        img = img.to(device)
        n = len(img)
    
        embedding = model.encode_image(img.half())
        embeddings.append(embedding.cpu().numpy())
features = np.vstack(embeddings)

np.save("assets/search/cifar10_features_nfeat512.npy",features.astype("float32"))