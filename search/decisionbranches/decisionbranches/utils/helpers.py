import numpy as np

#expects dict in format: {"n_feat":"nind"}
def generate_fidxs_variable(feat_cfg,feats,dtype="int32",seed=42):
    feat_idxs = []
    for n_feat,n_ind in feat_cfg.items():
        fidxs = generate_fidxs(n_feat,n_ind,feats,dtype,seed)
        feat_idxs.extend(fidxs.tolist())

    max_length = max(len(lst) for lst in feat_idxs)
    feat_idxs = np.array([lst + [-1]*(max_length - len(lst)) for lst in feat_idxs],dtype=dtype)
    feat_idxs = np.ma.masked_equal(feat_idxs, -1)
    return feat_idxs

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

