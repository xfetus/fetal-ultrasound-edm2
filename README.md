# xfetusEDM2 -- :baby: :brain: :robot: -- FetalUltrasoundEDM2

xfetusEDM2 provides tools for training and evaluating EDM2 diffusion models on the Open Fetal Planes Ultrasound dataset.
The repository includes utilities for training models, generating synthetic images, and evaluating performance using FID metrics.

## Getting Started

:nut_and_bolt: Installation
1. Install NVIDIA Drivers

Ensure that your system has compatible NVIDIA drivers installed.
```bash
sudo apt install nvidia-driver-550 #Update the NVIDIA Driver
sudo reboot # if in local machine reboot 
```

2. Create a Python Environment (using uv)

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 # Create a virtual environment at .venv.
source .venv/bin/activate #To activate the virtual environment
uv pip install -e ".[test,learning]" # Install the package in editable mode
uv sync
uv pip list --verbose #check versions
```

## :brain: Training the Model

To train the EDM2 model on the the fetal planes dataset first download the dataset from https://zenodo.org/records/3904280. Next, run the following command in the root directory of this repo:

```bash
torchrun --standalone --nproc_per_node=1 train_edm2.py \
            --outdir /OUTPUT_DIRECTORY \
            --data /DATASET_LOCATION \
            --batch 8 \
            --preset edm2-img512-s \
            --batch-gpu=8
```

where `DATASET_LOCATION` should be the root directory of the downloaded fetal planes dataset and `OUTPUT_DIRECTORY` is the location we will save our model checkpoints. This particular example trains on the small (s) version of the EDM2 architecture.

## 🖼 Generating Synthetic Images

Once our model is trained we generate 5k image per class. This can be done using the following bash code:

```bash
for class_idx in 0 1 2 3 4 5; do
        python generate_images.py \
                --preset=edm2-img512-s-guid-fid \
                --net_ckpt=./OUTPUT_DIRECTORY/training-state-0008519.pt \
                --gnet_ckpt=./OUTPUT_DIRECTORY/training-state-0001310.pt \
                --outdir=./OUTPUT_DIRECTORY/diffusion_samples_FETAL_cond_${class_idx} \
                --guidance 1.5 \
                --seeds=0-5000 \
                --class=${class_idx}
done
```

Generation require two network checkpoints (the first should be trained for longer than the second). In this example, we have set the first model to `training-state-0008519.pt` and the second model to `training-state-0001310.pt`, but whatever checkpoints can be used here, just make sure the `net_ckpt` has been trained for longer. The `guidance` flag controls the strength of the autoguidance and may need to be tuned for optimal performance. The `outdir` flag is where the generated images will be saved.

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
```
uv run pre-commit run -a
```
This ensures code formatting and linting checks pass.