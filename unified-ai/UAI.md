# UCL unified-ai pipelines

## Connect to vpn and Unified AI platform

1. Setup and connect to [UCL VPN](https://www.ucl.ac.uk/isd/services/get-connected/ucl-virtual-private-network-vpn)
2. Connect to https://kubeflow.arc-unified-ai.condenser.arc.ucl.ac.uk/

## Create a Unified AI Notebook

* Open the Unified AI platform and create a new notebook with a name of your choice and 2cpus and 4Gi of mem.
* Under Data Volumes, select scratch-volume. scratch-volume is required to store datasets, models, and outputs generated during your session.

## Using ghcr
Clone repo in `/home/jovyan`
```bash
git clone https://github.com/xfetus/fetal-ultrasound-edm2.git
```

## Using scratch-volume
Clone repo in `scratch-volume`
```bash
git clone https://github.com/xfetus/fetal-ultrasound-edm2.git
```
