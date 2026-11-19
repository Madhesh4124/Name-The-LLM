---
name: gcp-gpu-job
description: Use when launching, monitoring, resuming or tearing down GPU or CPU jobs on Google Cloud Compute Engine for this project.
---
# GCP GPU jobs

All values come from configs/gcp.env. Use `gcloud storage` for bucket I/O. Always get explicit user confirmation before create, start, stop or delete.

## Create a Spot GPU VM (verify flags with `gcloud compute instances create --help`)
```
source configs/gcp.env
gcloud compute instances create "$VM_NAME" \
  --project="$PROJECT_ID" --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --provisioning-model=SPOT --instance-termination-action=STOP \
  --maintenance-policy=TERMINATE \
  --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="${BOOT_DISK_GB}GB" --boot-disk-type=pd-balanced \
  --scopes=storage-rw \
  --metadata=install-nvidia-driver=True
```
Add an accelerator flag only if the chosen machine type requires it [verify for G2/A2].

## Job wrapper (scripts/run_job.sh, create if missing)
1. Resume from the latest checkpoint in $BUCKET/ckpt/<run_id>/ if present.
2. Run training inside tmux, writing a checkpoint to the bucket every N steps.
3. On finish: gcloud storage rsync artifacts/<exp>/ to $BUCKET/oof_test/<exp>/, then `sudo shutdown -h now`.
4. If the VM was preempted, restart it (with confirmation) and rerun the same command; it must resume.

## Pilot before full run
Run about 100 steps, record samples/s, compute GPU-hours = folds x epochs x N_train / throughput, multiply by the current spot price from the pricing calculator.

## Environment
- Match Kaggle image versions for torch, transformers, scikit-learn, lightgbm.
- Save the output of `pip freeze` to artifacts/<exp>/env.txt.

## Cloud Run Jobs fallback
If Compute Engine GPU quota is unavailable or delayed:
```
source configs/gcp.env
gcloud run jobs create "$VM_NAME" \
  --project="$PROJECT_ID" --region="$REGION" \
  --image=<TRAINING_IMAGE> \
  --gpu=1 --no-gpu-zonal-redundancy \
  --memory=32Gi --cpu=8 --task-timeout=6h --max-retries=3 \
  --set-env-vars="BUCKET=$BUCKET,EXP=<exp>" \
  --command="bash,scripts/run_job.sh"
```
This auto-grants 3×L4 per region without a manual quota request. Same checkpoint-to-GCS
and cost-safety rules apply. Get explicit user confirmation before creating or executing.

## Teardown
List instances, disks, and Cloud Run jobs; delete only confirmed items. Never delete the bucket.
