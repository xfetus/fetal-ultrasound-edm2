# UCL unified-ai pipelines

## 1. Connect to VPN and Unified AI Platform

1. Setup and connect to [UCL VPN](https://www.ucl.ac.uk/isd/services/get-connected/ucl-virtual-private-network-vpn)
2. Connect to https://kubeflow.arc-unified-ai.condenser.arc.ucl.ac.uk/

## 2. Create a Unified AI Notebook

* Open the Unified AI platform and create a new notebook with a name of your choice and default 2 CPUs, and 4Gi of memory.
* Under Data Volumes, select scratch-volume. This volume is required to store datasets, models, and outputs generated during your session.

## 3. Clone the Repository

### Using ghcr
Clone repo in `/home/jovyan`
```bash
git clone https://github.com/xfetus/fetal-ultrasound-edm2.git
```

### Using scratch-volume
Clone repo in `scratch-volume`
```bash
git clone https://github.com/xfetus/fetal-ultrasound-edm2.git
```

## kubectl Troubleshooting

### List trainjobs, jobs and pods
```bash
clear
kubectl get trainjobs,jobs,pods
```

### describe train jobs
```bash
kubectl describe trainjob <job-name>
```

### See tail of the generated log file
```bash
clear && tail -n 50 ../data-fetal-us-edm2/OUTPUT_DIRECTORY/log.txt
```

### List Jobs
```bash
kubectl get jobs
```

### Describe a Job.
```bash
kubectl describe job <job-name>
```

### Get traijobs
```bash
kubectl get trainjobs
```

### Delete trainjobs
```bash
kubectl delete trainjob <job-name>
kubectl delete trainjobs --all
```

### Delete a Job.
```bash
kubectl delete job <job-name>
```

### Delete all jobs
```bash
kubectl delete jobs --all
```

### Get the training runtime config

Get `clustertrainingruntime` and save it to [torch-distributed.yaml](torch-distributed.yaml)
```bash
kubectl get clustertrainingruntime torch-distributed -o yaml > torch-distributed.yaml
```


### Check your quota

Check quota usage with clusterqueue. Current usage is shown at the bottom, under the `status.flavorsUsage` field:
```bash
kubectl describe clusterqueue dev-shared
```
