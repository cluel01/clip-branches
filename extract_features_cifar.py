import os
import torch
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import clip
import pandas as pd
from PIL import Image
from src.utils import quantize_features_uniform
from src.models import CLIPModelWrapper

# Download and load the CIFAR10 dataset
print("##### Downloading and loading the CIFAR10 dataset... #####")
dataset = CIFAR10(root='./data/datasets/cifar', train=True, download=True, transform=transforms.ToTensor())
print("##### Dataset loaded successfully. #####")

# Initialize an empty list to store image paths and labels
image_dataset = []

# Iterate over the dataset and save each image in its corresponding class directory
print("##### Iterating over the dataset and saving images in corresponding class directories... #####")
for i, (image, label) in enumerate(dataset):
    # Define the directory name based on the class label
    lab = dataset.classes[label]
    label_dir = os.path.join('data/datasets/cifar/', lab)
    os.makedirs(label_dir, exist_ok=True)
    
    # Convert the tensor image back to PIL image
    pil_image = transforms.ToPILImage()(image)

    # Define the image file path
    image_path = os.path.join(label_dir, f'image_{i}.png')

    # Save the image
    pil_image.save(image_path)

    # Append the image path and label to the list
    image_dataset.append([os.path.join(lab, f'image_{i}.png'), lab])

print("##### All images have been saved. #####")

# Create a DataFrame from the image dataset
df_clip = pd.DataFrame(image_dataset, columns=["filename", "caption"])
print("##### DataFrame created with image paths and labels. #####")

# Save the first 1000 rows of the DataFrame to a CSV file
# df_clip.iloc[:1000].to_csv("data/cifar_dataset.csv", index=False)
df_clip.to_csv("data/datasets/cifar_dataset.csv", index=False)

print("##### First 1000 image paths and labels saved to data/datasets/cifar_dataset.csv #####")

#### Feature extraction #########

# Set the device to GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"##### Using device: {device} #####")

# Load the CLIP model
print("##### Loading the CLIP model... #####")
model = CLIPModelWrapper("ViT-B/32", device=device, freeze_backbone=True, custom_head=True,
                         distance="euclidean", embed_dim=32, hidden_dim=8192)
model.load_state_dict(torch.load("assets/search/data/clip_weights_v6.pth", map_location="cpu"))
preprocess = model.preprocess
print("##### Model loaded successfully. #####")

# Download and load the training data with preprocessing
print("##### Downloading and loading the training data... #####")
trainset = torchvision.datasets.CIFAR10(root='./data/datasets/cifar', train=True,
                                        download=True, transform=preprocess)
print("##### Training data loaded and preprocessed. #####")

# Only use the first 1000 images for training
# trainset = torch.utils.data.Subset(trainset, range(1000))

# Create a DataLoader for the training data
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=False, num_workers=2)
print("##### DataLoader created for the training data. #####")

# Set the model to evaluation mode
model.eval()
print("##### Model set to evaluation mode. #####")

# Initialize a list to store the embeddings
embeddings = []

# Extract features from the images in the training DataLoader
print("##### Extracting features from the images... #####")
with torch.no_grad():
    for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
        # Get the images from the data
        img = data[0]
        img = img.to(device)
        n = len(img)
        
        # Encode the images to get embeddings
        embedding = model.encode_image(img.half())
        
        # Append the embeddings to the list
        embeddings.append(embedding.cpu().numpy())

# Stack the embeddings into a numpy array
features = np.vstack(embeddings)
print("##### Features extracted successfully. #####")
print(f"##### Shape of features: {features.shape} #####")
print(f"##### Data type of features: {features.dtype} #####")

# Quantize the features uniformly
features = quantize_features_uniform(features)
print("##### Features quantized uniformly. #####")
print(f"##### Shape of quantized features: {features.shape} #####")
print(f"##### Data type of quantized features: {features.dtype} #####")

# Save the quantized features to a numpy file
np.save("assets/search/data/cifar10_features_nfeat32.npy", features.astype("uint8"))
print("##### Quantized features saved to assets/search/data/cifar10_features_nfeat32.npy #####")
