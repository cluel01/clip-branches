import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import numpy as np
import os
import pandas as pd
from PIL import Image
from clip import tokenize
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import re
import shutil

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CLIP_Dataset(Dataset):
    def __init__(self, root,labels_path, transform=None,subset_size = None,seed=32):
        np.random.seed(seed)
        self.root = root
        self.transform = transform
        self.seed = seed
        self.df = pd.read_csv(labels_path)

        if subset_size is not None:
            self.df = self.df.sample(subset_size,replace=False,random_state=seed)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        #load pil image
        img = Image.open(os.path.join(self.root, row['filename'])).convert('RGB')
        #### Shutterstock
        width, height = img.size
        crop_box = (0, 0, width, height - 30)
        img = img.crop(crop_box)
        ###################################

#        label = "shutterstock"
        label = row['caption']

        label = filter_caption(label)
       
        label = tokenize(label[:102]).squeeze(0) #102 is the character limit for CLIP

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.df)
    
    def show_image(self,idx):
        row = self.df.iloc[idx]
        print(row['caption'])
        img = Image.open(os.path.join(self.root, row['filename'])).convert('RGB')
        print(idx)
        plt.imshow(img)  
        plt.show()

class CLIP_DatasetV2(CLIP_Dataset):
    def __init__(self, root,labels_path, transform=None,seed=42):
        np.random.seed(seed)
        self.root = root
        self.transform = transform
        self.labels_path = labels_path
        self.seed = seed
        
        self.df = pd.read_csv(labels_path)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        #load pil image
        img = Image.open(os.path.join(self.root, row['filename'])).convert('RGB')
        t = eval(row['captions'])
        
        text = np.random.choice(t)

        label = tokenize(text).squeeze(0)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class PatternNetDataset(CLIP_Dataset):
    def __init__(self, root,labels_path, transform=None,seed=42,classes=None):
        np.random.seed(seed)
        self.root = root
        self.transform = transform
        self.labels_path = labels_path
        self.seed = seed
        
        self.df = pd.read_csv(labels_path)

        if classes is not None:
            idxs = []
            for c in classes:
                i = np.where(self.df["text"].str.contains(c))[0]
                idxs.extend(i)
            idxs = np.unique(idxs)
            self.df = self.df.iloc[idxs]



class MemoryMappedDataset(Dataset):
    def __init__(self, length=None,transform=None):
        """
        A PyTorch Dataset that uses a memory-mapped file to store RGB images.

        :param mmap_path: The file path to the memory-mapped file
        :param transform: Optional transform to be applied on a sample
        """
        # if not os.path.exists(mmap_path):
        #     raise FileNotFoundError(f"Memory-mapped file {mmap_path} does not exist.")
        
        # Open the memory-mapped file in read-only mode
        self.mmap_2021 = np.memmap("/home/jovyan/work/satellite_data/digital_orthophoto_nrw/datasets/nrw_2021_training_patches_1M_V2.mmap",shape=(1000000,3,224,224), dtype=np.uint8, mode='r')
        self.mmap_2022 = np.memmap("/home/jovyan/work/satellite_data/digital_orthophoto_nrw/datasets/nrw_2022_training_patches_1M_V2.mmap",shape=(1000000,3,224,224), dtype=np.uint8, mode='r')
        
        # Assuming the memory-mapped file stores images in CHW format (Channels, Height, Width)
        # Here, we must know the shape of the images and the number of images stored in the mmap
        self.image_shape = (3,224,224)  # Example image shape; this should be set correctly
        if length is None:
            self.N  = 2_000_000
        else:
            self.N = length
        
        #create a mapping that refers to both mmaps and the number of images in each

        # Reshape the mmap to an array of images
        #self.mmap = self.mmap.reshape((self.num_images,) + self.image_shape)
        
        # Transformations that should be applied to the images
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return self.N

    def __getitem__(self, idx):
        """
        Gets the image at a specified index.

        :param idx: Index of the image to retrieve
        """
        # Ensure the index is within the range of the dataset
        if idx >= self.N or idx < 0:
            raise IndexError("Index out of range")

        if (idx % 2) == 0:
            arr = self.mmap_2021[idx//2][:]
        else:
            arr = self.mmap_2022[idx//2][:] 

        # Retrieve the image from the memory-mapped array
        img = Image.fromarray(arr.transpose(1,2,0),"RGB")

        # Apply transformations
        if self.transform:
            img = self.transform(img)

        return img

    def show_image(self,idx):
        
        if (idx % 2) == 0:
            arr = self.mmap_2021[idx//2][:]
        else:
            arr = self.mmap_2022[idx//2][:] 
        img = Image.fromarray(arr.transpose(1,2,0),"RGB")
        print(idx)
        plt.imshow(img)  
        plt.show()

class NumpyArrayDataset(Dataset):
    def __init__(self, arr_path,captions_path, transform=None,seed=42):
        """
        A PyTorch Dataset that uses a memory-mapped file to store RGB images.

        :param mmap_path: The file path to the memory-mapped file
        :param transform: Optional transform to be applied on a sample
        """
        np.random.seed(seed)
        if not os.path.exists(arr_path):
            raise FileNotFoundError(f"Memory-mapped file {arr_path} does not exist.")
        
        # Open the memory-mapped file in read-only mode
        self.df = np.load(arr_path)

        self.captions = pd.read_csv(captions_path)
                
        # Transformations that should be applied to the images
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Gets the image at a specified index.

        :param idx: Index of the image to retrieve
        """
        # Ensure the index is within the range of the dataset
        if idx >= len(self.df) or idx < 0:
            raise IndexError("Index out of range")

        # Retrieve the image from the memory-mapped array
        img = Image.fromarray(self.df[idx],"RGB")
        cap = self.captions.iloc[idx]["captions"]
        t = eval(cap)
        text = np.random.choice(t)
        label = tokenize(text).squeeze(0)
        
        # Apply transformations
        if self.transform:
            img = self.transform(img)

        return img,label

    def show_image(self,idx):
        img = Image.fromarray(self.df[idx],"RGB")
        cap = self.captions.iloc[idx]["captions"]
        print(idx)
        print(cap)
        plt.imshow(img)  
        plt.show()

class CLIP_ImageFolder(Dataset):
    def __init__(self, root, transform=None,seed=42):
        self.dataset = ImageFolder(root)
        self.transform = transform
        self.seed = seed
        np.random.seed(seed)



    def __getitem__(self, idx):
        img,label = self.dataset[idx]
        text_label = self.dataset.classes[label]

        text_label = text_label.replace("_"," ")

        text_label = f"A photo of a {text_label}"
       
        text_label = tokenize(text_label).squeeze(0) #102 is the character limit for CLIP

        if self.transform is not None:
            img = self.transform(img)

        return img, label #text_label

    def __len__(self):
        return len(self.dataset)
    
    def show_image(self,idx):
        path,label = self.dataset.imgs[idx]
        text_label = self.dataset.classes[label]
        text_label = text_label.replace("_"," ")
        text_label = f"A photo of a {text_label}"
        print(text_label)
        img = Image.open(path).convert('RGB')
        print(idx)
        plt.imshow(img)  
        plt.show()

    def split_dataset(self,train_path,val_path,train_size):
        imgs = np.array(self.dataset.imgs)
        
        n_train = int(train_size * len(imgs))

        train_imgs = np.random.choice(len(imgs),size=n_train,replace=False)
        val_imgs = np.setdiff1d(np.arange(len(imgs)),train_imgs)

        train_imgs = imgs[train_imgs]
        val_imgs = imgs[val_imgs]

        for path,imgs in zip([train_path,val_path],[train_imgs,val_imgs]):
            for img in imgs:
                img_path = img[0]
                label = int(img[1])
                class_name = self.dataset.classes[label]
                new_dir = os.path.join(path,class_name)
                new_path = os.path.join(new_dir,os.path.basename(img_path))
                os.makedirs(new_dir,exist_ok=True)

                shutil.copy(img_path,new_path)
        
                

            






def filter_caption(input_string):
    # This regular expression pattern matches only alphanumeric characters and punctuation marks
    # from the regular English alphabet and ignores other characters
    pattern = r'[a-zA-Z0-9\s\.\,\!\?\:\;\'\"\(\)\[\]\{\}\-\/\@\#\$\%\^\&\*\_]'
    filtered_caption = ''.join(re.findall(pattern, input_string))

    filtered_caption = ' '.join(filtered_caption.split())
    return filtered_caption