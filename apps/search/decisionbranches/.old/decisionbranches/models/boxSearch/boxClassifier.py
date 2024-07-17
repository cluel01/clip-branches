from .boxSearch import BoxSearch

"""
sklearn compatible wrapper classifier for decision branch model
(only works for binary classification!!!)
"""
class BoxClassifier(BoxSearch):
    def __init__(self,tot_feat,n_feat=None,n_ind=None,max_feat="all",postTree=False,n_jobs=1,
                    cfg=None,feats=None,indices=None, verbose=True,debug=False,dtype="float64",dtype_int="int32",seed=42):
        '''
        tot_feat: total number of features
        n_feat: number of features per subset
        n_ind: number of feature subsets
        max_feat: number of feature subsets that are evaluated per iteration 
        postTree: if top-down construction on all features is en-/disabled
        n_jobs: number of cores to use
        cfg: config dictionary for specifiy config parameters of single decision branches 
        feats: predefined number of features to be used for subset generation
        indices: predefined feature subsets
        verbose: verbose output
        debug: debug output
        dtype: float dtype
        dtype_int: int dtype
        seed: random seed
        '''
        super(BoxClassifier, self).__init__(verbose=verbose,debug=debug,dtype=dtype,dtype_int=dtype_int,seed=seed)
        
        assert ((n_feat is not None) & (n_ind is not None)) or (indices is not None), "Missing args for index generation!"

        self.init_cfg = cfg
        self.postTree = postTree
        self.n_jobs = n_jobs

        if indices is not None:
            self.set_feat_idxs(feat_idxs=indices,tot_feat=tot_feat)
        else:
            self.build_indices(n_feat=n_feat,n_ind=n_ind,tot_feat=tot_feat,max_feat=max_feat,feats=feats)

    def fit(self,x,y,pos_label=1,**kwargs):

        self.load_data(x,y,pos_label)
        
        if self.postTree==False:
            if self.n_jobs > 1:
                self.train_boxSearch_parallel(n_jobs=self.n_jobs,**self.init_cfg)
            else:
                self.train_boxSearch(**self.init_cfg)
        else:
            if self.n_jobs > 1:
                self.train_boxSearch_parallel(n_jobs=self.n_jobs,**self.init_cfg)
            else:
                self.train_boxSearch(**self.init_cfg)
            self.train_postTree(**kwargs)
    

    def predict(self,x,box_ids="all"):
        return super(BoxClassifier,self).predict(x,box_ids)
