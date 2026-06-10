# FETAL_PLANES_DB: Common maternal-fetal ultrasound images (12,400 images in 2.1GB)

**Burgos-Artizzu, X.P., Coronado-Gutiérrez, D., Valenzuela-Alcaraz, B. et al. Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes. Sci Rep 10, 10200 (2020). https://doi.org/10.1038/s41598-020-67076-5**
### Data Description
A large dataset of routinely acquired maternal-fetal screening ultrasound images collected from two different hospitals by several operators and ultrasound machines. All images were manually labeled by an expert maternal fetal clinician (B.V-A.). Images were divided into 6 classes: four of the most widely used fetal anatomical planes (Abdomen, Brain, Femur and Thorax), the mother’s cervix (widely used for prematurity screening) and a general category to include any other less common image plane. Fetal brain images were further categorized into the 3 most common fetal brain planes (Trans-thalamic, Trans-cerebellum, Trans-ventricular) to judge fine grain categorization performance. The final dataset is comprised of over 12,400 images from 1,792 patients.

Images are in `./Images/*.png`

All information related with the images is in `FETAL_PLANES_DB_data` (provided both in csv and xlsx formats)

The dataset details are described in our open-acces paper: [Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes](https://rdcu.be/b47NX)


## Download dataset

Site and size
* https://zenodo.org/records/3904280
* `FETAL_PLANES_ZENODO.zip md5:2a5fcc2cefb789bcc0f6c1f73e0ea43f 	2.1 GB`


```bash
mkdir -p ~/scratch-volume/FETAL_PLANES_DB && mkdir -p ~/scratch-volume/FETAL_PLANES_DB/OUTPUT_DIRECTORY && cd ~/scratch-volume/FETAL_PLANES_DB
wget -c --content-disposition https://zenodo.org/records/3904280/files/FETAL_PLANES_ZENODO.zip?download=1
unzip FETAL_PLANES_ZENODO.zip && rm FETAL_PLANES_ZENODO.zip
```


If you find this dataset useful, please cite:

    @article{Burgos-ArtizzuFetalPlanesDataset,
      title={Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes},
      author={Burgos-Artizzu, X.P. and Coronado-Gutiérrez, D. and Valenzuela-Alcaraz, B. and Bonet-Carne, E. and Eixarch, E. and Crispi, F. and Gratacós, E.},
      journal={Nature Scientific Reports},
      volume={10},
      pages={10200},
      doi="10.1038/s41598-020-67076-5",
      year={2020}
    }




# FPUS23_Dataset: fetus phantom ultrasound dataset (15,728 images in 4.93Gb)
FPUS23, fetus phantom ultrasound dataset, can be used to identify (1) the correct diagnostic planes for estimating fetal biometric values, (2) fetus orientation, (3) their anatomical features, and (4) bounding boxes of the fetus phantom anatomies at 23 weeks gestation. The entire dataset is composed of 15,728 images.

```bash
wget -O FPUS23_Dataset.zip \
"https://drive.usercontent.google.com/download?export=download&confirm=t&id=1LL-r2hNiP6C190UBSE4v1FFCF3OQT9N3"
```

Prabakaran, Bharath Srinivas, Paul Hamelmann, Erik Ostrowski, and Muhammad Shafique. "FPUS23: an ultrasound fetus phantom dataset with deep neural network evaluations for fetus orientations, fetal planes, and anatomical features." IEEE Access 11 (2023): 58308-58317.
	* https://github.com/bharathprabakaran/FPUS23
	* google citations https://scholar.google.com/scholar?cites=4749675840285171641&as_sdt=2005&sciodt=0,5&hl=en 



# Fetal Abdominal Structures Segmentation Dataset Using Ultrasonic Images (1588 images,  1.1GB)
```bash
wget --content-disposition \
-O fetal_dataset.zip \
"https://data.mendeley.com/public-files/datasets/4gcpm9dsc3/files/89e74076-ff57-4e81-9634-4fc29c6128ff/file_downloaded"
```

## References
* https://data.mendeley.com/datasets/4gcpm9dsc3/1 
* https://www.kaggle.com/datasets/orvile/fetal-abdominal-structures-segmentation-dataset/data



# Maternal fetal ultrasound planes from low-resource imaging settings in five African countries (451 images in 42.8 MB)

```bash
wget -c --content-disposition \
"https://zenodo.org/api/records/7540448/files/Zenodo_dataset.tar.xz/content"
```


## References
* Preprint: https://arxiv.org/abs/2209.09610
* Journal article: https://www.nature.com/articles/s41598-023-29490-3



# Pre-trained Models

```bash
mkdir -p ~/scratch-volume/FETAL_PLANES_DB/models/sd-vae-ft-mse && cd ~/scratch-volume/FETAL_PLANES_DB/models/sd-vae-ft-mse
wget -4 -O config.json https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json
wget -4 -O diffusion_pytorch_model.safetensors https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors
```
