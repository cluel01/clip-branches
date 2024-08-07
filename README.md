# CLIP-Branches: Interactive Fine-Tuning for Text-Image Retrieval
This repository contains the source code for the paper *"CLIP-Branches: Interactive Fine-Tuning for Text-Image Retrieval"*. Our prototype is available via this [link](https://web.clip-branches.net/). 


## Installation
1. Clone the repository
```git clone https://github.com/cluel01/clip-branches.git``` 

2. Install the required packages
```pip install -r requirements.txt```

3. Install CLIP library:
```pip install git+https://github.com/openai/CLIP.git```

### Pre-requisites
Before actually running the application and submitting queries, you need to first run through the **offline preprocessing** phase as shown in the figure.

<img src="./img/framework.png" width="450" height="320">

This involves the following steps:
1. **Download Dataset**: Download image datasets of your choice. We provide a script to download the Shutterstock dataset used in our experiments. Please ensure to download the metadata first via the [link](https://drive.google.com/file/d/1mSNAL7u8y39O_fb66f38uLRm1zUnDH9O/view?usp=sharing). **Alternatively**, you can also use your own dataset or **CIFAR10** in order to speed up the preprocessing phase. For CIFAR10, you can jump to step 3, where we have provided a script to extract the features.

```python download_shutterstock.py```

2. **Preprocess Images**: Preprocess the images and labels and store them in a format that can be used by the CLIP model. Thereby, we bring the images and labels into a suitable format for the CLIP feature extractor and filter out incorrect images/text pairs. As an output, we get a CSV file that can be used in the next steps. An example is given for the Shutterstock dataset.

```python preprocess.py```

3. **Extract Features**: Extract image and text features using the CLIP model. The extracted features are stored in a file that can be used in the next steps. Make sure that you have downloaded our finetuned weighs for the CLIP model in advance via this [link](https://drive.google.com/file/d/1vwTrJbQntVuZpPazrpirS7n2ck-okrdg/view?usp=drive_link). Alternatively, you can also use the original CLIP model and weights. Depending on the size of the dataset and if you are using an GPU, this can take some time. We provide a script to extract the features for the Shutterstock dataset and CIFAR10.

```python extract_features_<shutterstock/cifar>.py```


### Running the Application
Now, we have to configure the services and start them. The architecture of our search platform is depicted in the following figure. It consists of three services: **Search**, **Data**, and **Web**. The Search service is responsible for the actual search functionality, the Data service provides the data for the search service, and the Web service is the user interface for the search service.

<img src="./img/arch.png" width="300" height="230">

The services are provided as Docker containers and can be started as follows:

4. **Configure Search App**: Configure the search service. Per dataset, you can define your own search procedure and store it in the folder `assets/search/config/`. An example configuration for the CIFAR10 dataset is given in the folder. Afterwards, build the docker image and start the search service.

```docker build -t search -f apps/search/Dockerfile apps/search/.```

```docker run -d --rm  -p 5000:80    -v ./assets/search/config:/usr/src/app/assets/config:ro  -v ./assets/search/data:/usr/src/app/assets/data:ro -v ./data/indexes:/usr/src/app/indexes --name search search```

5. **Configure Data App**: Configure the data service by creating a JSON config file per dataset in the folder `assets/data/`. An example configuration for the CIFAR10 dataset is given in the folder. Afterwards, build the docker image and start the data service.

```docker build -t data -f apps/data/Dockerfile apps/data/.```

```docker run --rm -d -p 5001:8000 -v ./assets/data/:/opt:ro -v ./assets/data/env:/usr/src/app/.env:ro -v  ./data/datasets:/mnt:ro --name data data```

6. **Configure Web App**:
Configure the web service by adjusting the file `assets/web/.env` to point to the search and data services as well as define the datasets. Afterwards, build the docker image and start the web service.

```docker build -t web -f apps/web/Dockerfile apps/web/.```

```docker run --rm -d -p 8888:3000 -v ./assets/web/env:/usr/src/app/.env:ro --name web web```

7. **Access the Web App and start search**: Access the web application via `http://localhost:8888` in your browser.

## Demo Video
[![Demo Video](https://img.youtube.com/vi/lepPM3zi0l8/0.jpg)](https://youtu.be/lepPM3zi0l8)

[Video](https://youtu.be/lepPM3zi0l8)



## Citation
If you find this work useful, please consider citing our paper:
```
@inproceedings{10.1145/3626772.3657678,
author = {L\"{u}lf, Christian and Lima Martins, Denis Mayr and Vaz Salles, Marcos Antonio and Zhou, Yongluan and Gieseke, Fabian},
title = {CLIP-Branches: Interactive Fine-Tuning for Text-Image Retrieval},
year = {2024},
isbn = {9798400704314},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3626772.3657678},
doi = {10.1145/3626772.3657678},
booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {2719–2723},
location = {Washington DC, USA},
series = {SIGIR '24}
}
```

## Use FAISS Index
Instead of using our kd-tree index for NN search of the text query, it is also possible to leverage state-of-the-art index structures for NN-search such as indexes coming from the FAISS library.
Configure kdt_textsearch in \<dataset\>.json for search service with the following:
```
        "type": "faiss",
        "index_dir": "indexes/{{ dataset.dataset_short }}/text_faiss",
        "index_file": "{{ dataset.dataset_short }}_flat_index.index",
```

