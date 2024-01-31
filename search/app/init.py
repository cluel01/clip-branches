import os
import pathlib
import numpy as np
import torch
from decisionbranches.utils.helpers import generate_fidxs,generate_fidxs_variable 
from searchers.db_searchengine.decisionbranch import DBSearchEngine,EnsembleSearchEngine
from searchers.boxnet.boxnet import BoxNetSearchEngine
from searchers.trees.decisiontree import DTreeSearchEngine
from searchers.trees.randomforest import RFSearchEngine
from searchers.nn_search import NNSearchEngine
from py_kdtree.kdtree import KDTree
from py_kdtree.kdtree_int import KDTree as KDTreeInt
from py_kdtree.treeset import KDTreeSet
from textsearcher.clip import CLIPModelWrapper
import faiss

def initialize_searcher(search_cfg):
    searchers = {}
    for s in search_cfg.keys():
        print(f"######## Load Dataset: {s} ##############")
        CFG = search_cfg[s]
        non_rare_feat = None
        non_rare_idxs = None

        print("Load Features")
        is_mmap = False
        if CFG["feature_file"].endswith(".npy"):
            features_arr = np.load(CFG["feature_file"])
        elif CFG["feature_file"].endswith(".pth"):
            features_arr = torch.load(CFG["feature_file"]).numpy()
        elif CFG["feature_file"].endswith(".mmap"):
            features = np.memmap(CFG["feature_file"], dtype=CFG["feature_mmap_dtype"], mode='r', shape=(CFG["size_dataset"],CFG["nfeat"]))
            is_mmap = True

        
        if CFG["features_inmemory"] == True:
            if is_mmap:
                print("Load features into memory from mmap")
                features = features[:].copy()
            else:
                print("Keep features in memory")
                features = features_arr
                del features_arr
        else:
            if not is_mmap:
                if "write_mmap" in CFG.keys():   
                    if CFG["write_mmap"]:
                        mmap_file = os.path.join(CFG["feature_mmap_path"],s+".mmap")
                        print("Write features to disk")
                        pathlib.Path(CFG["feature_mmap_path"]).mkdir(parents=True,exist_ok=True)
                        features = np.memmap(mmap_file,dtype='float64', mode='w+', shape=tuple(features_arr.shape))
                        features[:] = features_arr[:]
                        del features_arr

        if "non_rare_idxs" in CFG.keys():
            if CFG["non_rare_idxs"] is not None:
                if CFG["non_rare_idxs"] ==  "auto":
                    print("Sample non rare idxs automatically!")
                    if "non_rare_idxs_num" in CFG.keys():
                        n_nonrare = CFG["non_rare_idxs_num"]
                    else:
                        n_nonrare = 1000
                    non_rare_idxs = np.random.choice(np.arange(len(features)),size=n_nonrare,replace=False)
                    non_rare_idxs.sort()
                    non_rare_feat = features[non_rare_idxs]
                    non_rare_feat = non_rare_feat.astype(CFG["kdt_search"]["dtype"])
                    print("N non rares:",len(non_rare_feat))

        ##### Search #########
        if "kdt_search" in CFG.keys():
            cfg = CFG["kdt_search"]
            if cfg["index_file"] is not None:
                    indexes = np.load(cfg["index_file"])
            else:
            
                if cfg["fidx_cfg"]["nfeat"] == "variable":
                    indexes =  generate_fidxs_variable(feat_cfg=cfg["fidx_cfg"]["cfg"],feats=np.arange(features.shape[1]),seed=CFG["seed"])
                    indexes = [idxs.compressed().tolist() for idxs in indexes ]
                else:
                    indexes =  generate_fidxs(n_feat=cfg["fidx_cfg"]["nfeat"],n_ind=cfg["fidx_cfg"]["nind"],feats=np.arange(features.shape[1]),seed=CFG["seed"])
                

            print("Load KDtree search")
            kdt_search = KDTreeSet(indexes,path=cfg["index_dir"],leaf_size=cfg["leafsize"],dtype=cfg["dtype"],verbose=False)
            print("Create Kdtree indexes")
            if CFG["features_inmemory"]:
                kdt_search.fit(features,mmap=False,build_inmemory=cfg["build_inmemory"])
            elif CFG["features_inmemory"] == "subset":
                kdt_search.fit(features,mmap=False,build_inmemory=cfg["build_inmemory"])
            else:
                kdt_search.fit(features,mmap=True,build_inmemory=cfg["build_inmemory"])
        else:
            print("No KDtree for search given!")
            kdt_search = None

        if "text_searcher" in CFG.keys():
            cfg = CFG["text_searcher"]
            if cfg["model_type"] == "CLIP":
                print("Load CLIP model")
                text_model = CLIPModelWrapper(**cfg["clip_cfg"])
                if cfg["model_weights_path"] is not None:
                    text_model.load_state_dict(torch.load(cfg["model_weights_path"],map_location=torch.device('cpu')))
                text_model.eval()

            print("Load KDtree text search")
            cfg = CFG["kdt_textsearch"]
            feat_sub = None
            if "type" in cfg.keys() and cfg["type"] == "treeset":
                kdt_text = None
                print("No KDtree text indexes created! Use treeset instead")
            elif "type" in cfg.keys() and cfg["type"] == "faiss":
                print("Use FAISS index")
                fname = os.path.join(cfg["index_dir"],cfg["index_file"])
                if os.path.isfile(fname):
                    kdt_text = faiss.read_index(fname,faiss.IO_FLAG_MMAP) #faiss.read_index(fname,faiss.IO_FLAG_MMAP) 
                else:
                    kdt_text = faiss.IndexFlatL2(features.shape[1])
                    kdt_text.add(features)
                    # quantizer = faiss.IndexFlatL2(features.shape[1])  # the quantizer
                    # kdt_text = faiss.IndexIVFFlat(quantizer, features.shape[1], 100, faiss.METRIC_L2)  # 100 clusters
                    # kdt_text.train(features)
                    # kdt_text.add(features)
                    os.makedirs(cfg["index_dir"],exist_ok=True)
                    faiss.write_index(kdt_text,fname)
            else:
                f = features
                if "fidx_cfg" in cfg.keys():
                    if cfg["fidx_cfg"]["nfeat"] < features.shape[1]:
                        feat_sub = generate_fidxs(n_feat=cfg["fidx_cfg"]["nfeat"],n_ind=cfg["fidx_cfg"]["nind"],feats=np.arange(features.shape[1]),seed=CFG["seed"]).reshape(-1)
                        f = features[:,feat_sub]
                    
                if not cfg["dtype"] == "uint8":
                    kdt_text = KDTree(leaf_size=cfg["leafsize"],path=cfg["index_dir"],dtype=cfg["dtype"],inmemory=cfg["inmemory"],verbose=False)
                else:
                    kdt_text = KDTreeInt(leaf_size=cfg["leafsize"],path=cfg["index_dir"],dtype=cfg["dtype"],inmemory=cfg["inmemory"],verbose=False)
                if kdt_text.tree is None:
                    print("Create Kdtree text index")
                    kdt_text.fit(f,build_inmemory=cfg["build_inmemory"])

        else:
            kdt_text = None
            text_model = None

        ######  Decisionbranches ########
        print("Load DB models")
        db_models = {}
        idx_cfg = CFG["kdt_search"]
        for s_name,s_cfg in CFG["searcher"].items():
            if s_name.startswith("db"):
                search_inst = DBSearchEngine(CFG["nfeat"],indexes,s_cfg["db_cfg"],s_cfg["njobs"],CFG["seed"],
                                            s_cfg["dtype"],s_cfg["min_box_size"])
            elif s_name.startswith("ens"):
                search_inst = EnsembleSearchEngine(CFG["nfeat"],indexes,
                                                    s_cfg["ens_nestimators"],s_cfg["db_cfg"],s_cfg["njobs"],CFG["seed"],s_cfg["dtype"])
            elif s_name.startswith("dtree"):
                search_inst = DTreeSearchEngine(s_cfg["dtree_cfg"],CFG["seed"],s_cfg["dtype"])
            elif s_name.startswith("rf"):
                search_inst = RFSearchEngine(s_cfg["rf_cfg"],CFG["seed"],s_cfg["dtype"])
            elif s_name.startswith("nn"):
                search_inst = NNSearchEngine(s_cfg["nn_k"],CFG["seed"],idx_cfg["dtype"])
            elif s_name.startswith("boxnet"):
                search_inst = BoxNetSearchEngine(CFG["nfeat"],indexes,s_cfg["nboxes"],s_cfg["scaling"],CFG["seed"],cfg=s_cfg["cfg"],dtype=s_cfg["dtype"])
            db_models[s_name] = search_inst

        searchers[s] = {"features":features,"kdt_search":kdt_search,"searcher":db_models,
                        "non_rare_feat": non_rare_feat,"non_rare_idxs":non_rare_idxs,"seed":CFG["seed"],"kdt_text":kdt_text,"text_model":text_model,"cfg":CFG,
                        "feat_sub":feat_sub}
        
    return searchers
    
