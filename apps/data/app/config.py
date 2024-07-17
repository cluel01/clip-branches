import os
from typing import List, Tuple, Union
from pydantic import BaseSettings, BaseModel

class DatasetsList(BaseModel):
    datasets: List[str]


class Settings(BaseSettings):
    """Settings read from the .env file in the parent directory"""
    # endpoint_url: str
    # env_test: str
    path_cifar: str
    path_unsplash_lite: str
    path_unsplash_lite_512: str
    path_shutterstock: str
    path_laion_1m: str
    path_laion_262m: str
    #path_laion_262m_full: str
    assets_path: str
    class Config:
        env_prefix = ''
        env_file = ".env"
        env_file_encoding = 'utf-8'

def set_performance_environment_variables():
    os.environ["VSI_CACHE"] = "TRUE"

    
def create_filepath_dict(settings):
    filepath_dict = {}
    for key, value in settings.__dict__.items():
        if key.startswith("path_"):
            filepath_dict[key.replace("path_", "")] = value
    return filepath_dict