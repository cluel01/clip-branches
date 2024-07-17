import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  DataLoader,SubsetRandomSampler
import numpy as np
import clip
from src.loss import MMD_loss,KoLeoLoss
from src.validate import validate

def get_subset_random_sampler(total_size,subset_size):
    indices = np.random.choice(total_size, subset_size, replace=False)
    return SubsetRandomSampler(indices)

def train(model,device,batch_size,train_loader,val_loader,optimizer,lambda_,embed_dim,
          n_epochs=10,loss="softmax",aux_loss_type="mmd",model_path="model.pt",train_subset_size=None,seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    if aux_loss_type == "mmd":
        aux_loss_fn = MMD_loss()
        sampl_pop = torch.rand(batch_size,embed_dim).to(device) * 2 - 1
    elif aux_loss_type == "koleo":
        aux_loss_fn = KoLeoLoss()


    
    # Train the model
    best_acc = 0.0
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_cont_loss = 0.0
        running_aux_loss = 0.0
        c = 0

        if train_subset_size is not None:
            sampler = get_subset_random_sampler(len(train_loader.dataset),train_subset_size)
            train_loader = DataLoader(train_loader.dataset, sampler=sampler,batch_size=batch_size,
                                      shuffle=False, num_workers=train_loader.num_workers, pin_memory=True)
            

        for i, data in enumerate(train_loader, 0):
            # Define the loss function and optimizer
            model.train()
            optimizer.zero_grad()
            # get the inputs; data is a list of [inputs, labels]
            img, text = data
            img = img.to(device)
            text = text.to(device)
            n = len(img)

            with torch.autocast(device_type="cuda"):
                logits_img,logits_text,image_features,text_features = model(img,text)

                if aux_loss_type == "mmd":
                    aux_loss = (aux_loss_fn(image_features,sampl_pop[:n]) + aux_loss_fn(text_features,sampl_pop[:n]))/2
                elif aux_loss_type == "koleo":
                    aux_loss = (aux_loss_fn(image_features) + aux_loss_fn(text_features))/2
                #print("Num masked: ",(~mask).sum())
                    
                if loss == "softmax":    
                    ground_truth = torch.arange(n).to(device)
                    cont_loss = (loss_img(logits_img ,ground_truth) + loss_txt(logits_text,ground_truth))/2 
                    total_loss =  cont_loss + lambda_ * aux_loss
                    total_loss.backward()
                elif loss == "sigmoid":
                    ground_truth = (2 * torch.eye(n) - torch.ones(n)).to(device)
                    cont_loss = -torch.sum((F.logsigmoid(ground_truth*logits_img))) / n
                    total_loss =  cont_loss + lambda_ * aux_loss
                    total_loss.backward()
                if device == "cpu":
                    optimizer.step()
                else : 
                    #model._convert_models_to_fp32()
                    optimizer.step()
                    #model._convert_models_to_fp16()

            running_cont_loss += cont_loss.item()
            running_loss += total_loss.item()
            running_aux_loss += aux_loss.item()
            c += n

            

            
        train_acc = validate(model,train_loader,device,bs=batch_size,verbose=False)
        val_acc = validate(model,val_loader,device,bs=batch_size,verbose=False)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),model_path)

        print("Epoch: ",epoch)
        print("Loss: ",running_loss/c)
        print("Cont Loss: ",running_cont_loss/c)
        print("Aux Loss: ",running_aux_loss/c)
        print("Train Accuracy: ",train_acc)       
        print("Validation Accuracy: ",val_acc)
        print("------------------------------------------------------")