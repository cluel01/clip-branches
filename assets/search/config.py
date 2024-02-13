"""
Configurations variables for the search service.
"""
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Settings for the search service."""

    app_name: str = "Search Service"

    class Config:
        cfg = {
                "dataset_name": {
                        "feature_file": "assets/features.npy",
                        "feature_mmap_path": "indexes/features",
                        "feature_mmap_dtype": "uint8",
                        "size_dataset": 262_563_993, 
                        "features_inmemory": True,
                        "non_rare_idxs": "auto",
                        "non_rare_idxs_num": 100_000,#30_000,
                        "nfeat":32,
                        "seed": 42,
                        "kdt_search":{
                                "index_file": None,
                                "index_dir": "indexes/laion_var",
                                "leafsize": 4500,
                                "dtype":"uint8",
                                "build_inmemory":True,
                                "fidx_cfg": {"nfeat": "variable","cfg":{4:64,6:32,8:8}},
                        },
                        "kdt_textsearch":{
                                "type": "kdtree",
                                "index_file": None,
                                "index_dir": "indexes/dataset",
                                "leafsize": 4500,
                                "dtype":"uint8",
                                "inmemory":False,
                                "build_inmemory":True,
                                "fidx_cfg": {"nfeat":32,"nind":1},
                                "stop_leaves": 3000,
                        },

                        "text_searcher":{
                                "model_weights_path":"assets/clip_weights.pth",
                                "model_type": "CLIP",
                                "clip_cfg": {"model":"ViT-B/32","device":"cpu","train":False,"custom_head":True,"embed_dim":32,"hidden_dim":8192},
                        },
                        "searcher": {
                                "db": {
                                        "display_name": "Decisionbranches Demo",
                                        "db_cfg":{"max_nbox":30,"top_down":False,"max_evals":"all","stop_infinite":True,"postTree":True,"min_pts":"auto"},
                                        "njobs": 1,
                                        "dtype":"float64",
                                        "min_box_size": None,
                                },

                                "ens": {
                                        "ens_nestimators":25,
                                        "db_cfg":{"max_nbox":30,"top_down":False,"max_evals":0.5,"stop_infinite":False,"postTree":False,"min_pts":"auto"},
                                        "njobs": None,
                                        "display_name": "Decisionbranches Ensemble (25 trees)",
                                        "dtype":"float64",
                                },
                                "dtree": {
                                        "dtree_cfg":{},
                                        "display_name": "Decision Tree",
                                        "dtype":"float32",

                                },
                                "rf": {
                                        "rf_cfg": {"n_estimators":25},
                                        "display_name": "Random Forest (25t)",
                                        "dtype":"float32",
                                },
                        }

                  },

        }

