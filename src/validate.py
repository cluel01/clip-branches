import torch
import numpy as np
from tqdm import tqdm


def validate(model,val_loader, device, transform=None, bs=256, verbose=True,seed=42):
    torch.manual_seed(seed)
    
    model.eval()
    model.to(device)

    with torch.no_grad():
        total_correct = 0
        total = 0

        if verbose:
            val_loader = tqdm(val_loader)
        for data in val_loader:
            imgs = data[0].to(device)
            labels = data[1].to(device)

                
            n = imgs.shape[0]
            logits = model(imgs, labels)
            logits_imgs,logits_text = logits[0],logits[1]
            probs = logits_imgs.softmax(dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)

            # Update overall accuracy
            total_correct += (preds == np.arange(len(preds))).sum()
            total += n

    # Calculate total and per-class accuracies
    total_acc = total_correct / total

    if verbose:
        print("Total Accuracy: ", total_acc)
    return total_acc
    #return total_acc, class_accuracies

def validate_pca(model,val_loader, device,pca, transform=None, bs=256, verbose=True,seed=42):
    torch.manual_seed(seed)
    
    model.eval()
    model.to(device)

    with torch.no_grad():
        total_correct = 0
        total = 0

        if verbose:
            val_loader = tqdm(val_loader)
        for data in val_loader:
            imgs = data[0].to(device)
            labels = data[1].to(device)

                
            n = imgs.shape[0]
            image_features = model.encode_image(imgs)
            text_features = model.encode_text(labels)

            image_features = pca.transform(image_features.cpu().numpy())
            text_features = pca.transform(text_features.cpu().numpy())

            image_features = torch.from_numpy(image_features).to(device)
            text_features = torch.from_numpy(text_features).to(device)

            #normalize features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            logit_scale = model.logit_scale.exp()
            logits_imgs = logit_scale * image_features @ text_features.t()

            probs = logits_imgs.softmax(dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)

            # Update overall accuracy
            total_correct += (preds == np.arange(len(preds))).sum()
            total += n

    # Calculate total and per-class accuracies
    total_acc = total_correct / total

    if verbose:
        print("Total Accuracy: ", total_acc)
    return total_acc
    #return total_acc, class_accuracies
            