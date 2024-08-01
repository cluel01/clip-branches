import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip
from clip.model import CLIP
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode as IMode

_MODELS = {
"ViT-B/32":{"embed_dim":512,"image_resolution":224, "vision_layers":12, "vision_width":768, "vision_patch_size":32,
            "context_length":77,"vocab_size":49408,"transformer_width":512,"transformer_heads":8,"transformer_layers":12},
"ViT-B/16":{"embed_dim":512,"image_resolution":224, "vision_layers":12, "vision_width":768, "vision_patch_size":16,
            "context_length":77,"vocab_size":49408,"transformer_width":512,"transformer_heads":8,"transformer_layers":12},
}

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=IMode.BICUBIC),
        CenterCrop(n_px),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class L2Normalize(nn.Module):
    def __init__(self):
        super(L2Normalize, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2, dim=1)

class CLIPModelWrapper(nn.Module):
    def __init__(self, model, device,train=True,custom_head=True,embed_dim=128,hidden_dim=2048,
                 freeze_backbone=False,distance='cosine',dtype=torch.float32,mode=None,load_weights=False):
        super().__init__()
        # m,p = clip.load(model, device="cpu", jit=False) #workaround to load float32 weights on gpu
        m = CLIP(**_MODELS[model])

        if model == "ViT-B/32" and load_weights:
            print("Loading ViT-B/32 pretrained weights")
            m.load_state_dict(torch.load("assets/data/ViT-B-32.pt", map_location="cpu"))

        self.custom_head = custom_head
        self.device = device
        self.freeze_backbone = freeze_backbone
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.distance = distance
        self.dtype = dtype
        self.mode = mode
        
        self.preprocess = _transform(224)

        if distance == "euclidean":
            m.logit_scale = nn.Parameter(torch.tensor(np.log(1/0.07)).float() )
        
        # m.uni_scale = nn.Parameter(torch.ones(1)).float()

        self.model = m.to(device)

        

        if custom_head:
            self.head_vis = nn.Sequential(
                nn.Linear(in_features=512, out_features=hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=hidden_dim, out_features=embed_dim, bias=True),
                L2Normalize()
            ).to(device)
            self.head_text = nn.Sequential(
                nn.Linear(in_features=512, out_features=hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=hidden_dim, out_features=embed_dim, bias=True),
                L2Normalize()
            ).to(device)

        if train and not freeze_backbone:
            for params in self.model.parameters():
                params.requires_grad_(True)
        else:
            for params in self.model.parameters():
                params.requires_grad_(False)
            self.model.logit_scale.requires_grad_(True)
            # self.model.uni_scale.requires_grad_(True)
            
            if custom_head and not train:
                for params in self.head_vis.parameters():
                    params.requires_grad_(True)
                for params in self.head_text.parameters():
                    params.requires_grad_(True)

    def forward(self, imgs,text):
        image_features = self.model.encode_image(imgs).to(self.dtype)
        text_features = self.model.encode_text(text).to(self.dtype)

        if self.custom_head:
            image_features = self.head_vis(image_features)
            text_features = self.head_text(text_features)
        else:
            # normalized features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # scale for high-dim
        # uni_scale = self.model.uni_scale#.exp().to(self.dtype)
        # uni_image_features = image_features * uni_scale
        # uni_text_features = text_features * uni_scale

        logit_scale = self.model.logit_scale.exp().to(self.dtype)
        
        if self.distance == 'cosine':
            # cosine similarity as logits
            logits_per_image = (logit_scale * image_features @ text_features.t()).to(self.dtype)
            logits_per_text = logits_per_image.t()
        elif self.distance == 'euclidean':
            logits_per_image = logit_scale * -torch.cdist(image_features, text_features, p=2)
            logits_per_text = logits_per_image.t()
        # shape = [global_batch_size, global_batch_size]


        return logits_per_image, logits_per_text, image_features,text_features#uni_image_features, uni_text_features
    
    def encode_image(self, x,mode="both"):
        if self.mode is not None:
            mode = self.mode

        if mode == "both" or mode == "clip":
            x = self.model.encode_image(x)
        
        if self.custom_head and (mode == "both" or mode == "head"):
            x = self.head_vis(x.float())
            #x = x * self.model.uni_scale
        return x
    
    def encode_text(self, x,mode="both",tokenize=False):
        if tokenize:
            x = clip.tokenize([x]).to(self.device)

        if self.mode is not None:
            mode = self.mode

        if mode == "both" or mode == "clip":
            x = self.model.encode_text(x)

        if self.custom_head and (mode == "both" or mode == "head"):
            x = self.head_text(x.float())
            #x = x * self.model.uni_scale
        return x

    def _convert_models_to_fp16(self):
        clip.model.convert_weights(self.model)

    def _convert_models_to_fp32(self):
        for p in self.model.parameters(): 
            p.data = p.data.float() 
            if p.grad is not None:
                p.grad.data = p.grad.data.float() 