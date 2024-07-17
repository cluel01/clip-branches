import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLearningLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLearningLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Assume batch contains pairs of (z_i, z_j) for each example, which are the embeddings
        from two different models of the same object in different modalities.
        We need to pull z_i, z_j together while pushing apart from other examples in the batch.
        """
        batch_size = z_i.size(0)

        # Normalize the representations along the feature dimension
        z_i_norm = F.normalize(z_i, p=2, dim=1)
        z_j_norm = F.normalize(z_j, p=2, dim=1)

        # Concatenate the representations
        representations = torch.cat([z_i_norm, z_j_norm], dim=0)

        # Compute the similarity matrix
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        # Scale the similarity with the temperature
        similarity_matrix /= self.temperature

        # Create the labels for the contrastive learning
        labels = torch.range(0, 2 * batch_size - 1, step=2).long().to(z_i.device)
        labels = torch.cat((labels, labels + 1), dim=0)

        # Use the InfoNCE loss
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss
    
class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss
    
class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search
    https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py"""

    def __init__(self,normalize=True,eps=1e-8,loss_type="standard"):
        super().__init__()
        self.eps = eps
        self.pdist = nn.PairwiseDistance(2, eps=eps)
        self.normalize = normalize
        self.loss_type = loss_type

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, X):
        with torch.cuda.amp.autocast(enabled=False):
            if self.normalize:
                X = F.normalize(X, eps=self.eps, p=2, dim=-1)
            I = self.pairwise_NNs_inner(X)  # noqa: E741
            distances = self.pdist(X, X[I])  # BxD, BxD -> B

            if self.loss_type == "standard":
                loss = -torch.log(distances + self.eps).mean()
            elif self.loss_type == "batch":
                loss = - torch.log(len(X) * distances + self.eps).mean()
        return loss