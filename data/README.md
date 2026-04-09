# FETAL_PLANES_DB: Common maternal-fetal ultrasound images

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


## Pre-trained Models

```bash
mkdir -p ~/scratch-volume/FETAL_PLANES_DB/models/sd-vae-ft-mse && cd ~/scratch-volume/FETAL_PLANES_DB/models/sd-vae-ft-mse
wget -4 -O config.json https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json
wget -4 -O diffusion_pytorch_model.safetensors https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors
```
