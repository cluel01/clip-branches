import numpy as np

#Temporary method since the indices are intended to be generated beforehand manually
def generate_fidxs(n_feat,n_ind,feats,dtype="int32",seed=42):
    np.random.seed(seed)

    feat_idxs = np.empty((0,n_feat),dtype=dtype)
    while len(feat_idxs) != n_ind:
        f_idx = []
        f = feats.copy().tolist()
        for _ in range(n_feat):
            r_idx = int(np.random.choice(np.arange(len(f)),1))
            f_idx.append(f.pop(r_idx))
        f_idx = np.sort(f_idx)
        feat_idxs = np.vstack([feat_idxs,f_idx])
        feat_idxs = np.unique(feat_idxs,axis=0)
        
    return feat_idxs

