# xfetusEDM2 -- :baby: :brain: :robot: -- FetalUltrasoundEDM2

xfetusEDM2 provides tools for training and evaluating EDM2 diffusion models on the Open Fetal Planes Ultrasound dataset.
The repository includes utilities for training models, generating synthetic images, and evaluating performance using FID metrics.

## Getting Started

:nut_and_bolt: Installation

* Install NVIDIA Drivers. Ensure that your system has compatible NVIDIA drivers installed.
```bash
sudo apt install nvidia-driver-550 #Update the NVIDIA Driver
sudo reboot # if in local machine reboot
```
* Check PyTorch and supported CUDA version https://pypi.org/project/torch/#history
:warning: PyTorch (2.11.0, released on Mar 23, 2026); CUDA versions available (CUDA 12.6, CUDA 12.8, CUDA 13.0 (stable))
:warning: PyTorch (2.10.0, released on Jan 21, 2026); CUDA versions available (CUDA 12.6, CUDA 12.8)


* Create a Python Environment (using uv)
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
uv venv --python 3.11 # Create a virtual environment at .venv.
source .venv/bin/activate #To activate the virtual environment
uv sync --extra test --extra learning
uv pip list --verbose #check versions
```

* Launch Jupyter locally
```bash
uv run jupyter notebook
```

* pre-commit hooks
```bash
#Generate the baseline file
mkdir -p .github
detect-secrets scan > .github/.secrets.baseline
# run pre-commit hooks
uv run pre-commit run -a
```

## :brain: Training the Model

To train the EDM2 on the small (s) version of the EDM2 architecture for the fetal planes dataset first download the dataset from https://zenodo.org/records/3904280 as shown in [data](data).
Then to train xxs, s-sized model (`edm2-img512-xxs`,  `edm2-img512-s`,) for ImageNet-512 using 1 or 8 GPUs (`--batch-gpu=1`, `--batch-gpu=8`), for example, run the following command in the root directory of this repo:

```bash
source .venv/bin/activate #To activate the virtual environment

torchrun --standalone --nproc_per_node=1 train_edm2.py \
            --outdir ~/scratch-volume/data-fetal-us-edm2/OUTPUT_DIRECTORY \
            --data ~/scratch-volume/data-fetal-us-edm2/FETAL_PLANES_DB \
            --fpus23 ~/scratch-volume/data-fetal-us-edm2/FPUS23 \
            --african ~/scratch-volume/data-fetal-us-edm2/AfricanDataset/Zenodo_dataset \
            --fetal-abdomen ~/scratch-volume/data-fetal-us-edm2/FetalAbdominalSegmentation/IMAGES \
            --batch 1 \
            --preset edm2-img512-xxs \
            --batch-gpu=1
```

where `DATASET_LOCATION` should be the root directory of the downloaded fetal planes dataset and `OUTPUT_DIRECTORY` is the location we will save `log.txt`, `stats.json` and our model checkpoints (e.g. `network-snapshot-0000000-0.050.pkl`, `network-snapshot-0000000-0.100.pkl`, etc ).

## 🖼 Generating Synthetic Images

Once our model is trained we generate 5k image per class. This can be done using the following bash code:

```bash
for class_idx in 0 1 2 3 4 5; do
        python generate_images.py \
                --preset=edm2-img512-s-guid-fid \
                --net_ckpt=./OUTPUT_DIRECTORY/training-state-xxxxxxx.pt \
                --gnet_ckpt=./OUTPUT_DIRECTORY/training-state-yyyyyyy.pt \
                --outdir=./OUTPUT_DIRECTORY/diffusion_samples_FETAL_cond_${class_idx} \
                --guidance 1.5 \
                --seeds=0-5000 \
                --class=${class_idx}
done
```

Generation require two network checkpoints (the first should be trained for longer than the second). In this example, the checkpoints `training-state-xxxxxxx.pt` and `training-state-yyyyyyy.pt` contain the weight of the diffusion models, with xxxxxxx and yyyyyyy corresponding to the number of training steps. In our result we set the first model to `training-state-0008519.pt` and the second model to `training-state-0001310.pt`, but whatever checkpoints can be used here, just make sure the `net_ckpt` has been trained for longer. The `guidance` flag controls the strength of the autoguidance and may need to be tuned for optimal performance. The `outdir` flag is where the generated images will be saved.

## 📊 Evaluating Model Performance (FID)

Finally, to measure the FID of the generated images you can use the following command:

```bash
python fid_measurement.py \
            --real_root ./DATASET_LOCATION \
            --csv_file ./DATASET_LOCATION/FETAL_PLANES_DB_data.csv \
            --fake_root ./OUTPUT_DIRECTORY/ \
            --split test \
            --batch_size 32 \
            --device cuda
```

where the `fake_root` flag is where generated images are saved.


## 🤝 Contributing

We welcome contributions from the community. Before submitting a PR:
```bash
uv run pre-commit run -a
```
This ensures code formatting and linting checks pass.


## Clone repository
You need to [authorize a personal access token for use with single sign-on](https://docs.github.com/en/enterprise-cloud@latest/authentication/authenticating-with-single-sign-on/authorizing-a-personal-access-token-for-use-with-single-sign-on)
```bash
git clone https://github.com/xfetus/fetal-ultrasound-edm2.git
```

## Huggingface

**Model Weights:** We also provided access to the model weights for both our large EDM2-XL mode and smaller EDM2-S model [here](https://huggingface.co/harveymannering/ultrasound-edm2). The minimal code needed to use these weight can be found in the model card and require dependancies from the [EDM2 repo](https://github.com/NVlabs/edm2) in order to run.

**Generated Images:** The 30k generated images can also be found on huggingface and can be downloaded from [here](https://huggingface.co/datasets/harveymannering/ultrasound_images_diffusion). Class indexes in the dataset correspond to the following labels:
- 0: 'Other'
- 1: 'Maternal cervix'
- 2: 'Fetal abdomen'
- 3: 'Fetal brain'
- 4: 'Fetal femur'
- 5: 'Fetal thorax'


## :scroll: Article
> Harvey Mannering, Yilin Zhang, Ziao Liu, Zhiwu Huang, Jacqueline Matthew, Miguel Xochicale. **"A Foundational EDM2-Based Generative Model for High-Resolution Synthetic Fetal Ultrasound Imaging from Open Datasets."** arXiv preprint arXiv:2608.05471 (2026). Published in the 30th UK Conference on Medical Image Understanding and Analysis, MIUA'26 Short paper track. Dublin, Irland. 20th - 22nd July 2026. [[Github-repository]](https://github.com/xfetus/fetal-ultrasound-edm2); 
[[arXiv-preprint]](https://arxiv.org/abs/2608.05471); 

<details>

<summary>See bibtex to cite paper:</summary>

```
@misc{mannering2026foundationaledm2basedgenerativemodel,
      title={A Foundational EDM2-Based Generative Model for High-Resolution Synthetic Fetal Ultrasound Imaging from Open Datasets}, 
      author={Harvey Mannering and Yilin Zhang and Ziao Liu and Zhiwu Huang and Jacqueline Matthew and Miguel Xochicale},
      year={2026},
      eprint={2608.05471},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2608.05471}, 
}
``` 

</details>