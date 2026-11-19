---
trigger: always_on
---
# Compute and cost safety (GCP)

- NEVER run a gcloud command that creates, resizes, starts, stops or deletes a resource without explicit user confirmation in chat, even if a workflow step is marked turbo.
- Read project, zone and bucket from configs/gcp.env. Never hardcode them. Never print or commit credentials, service-account keys or the Kaggle token.
- Before any GPU launch: show GPU-hours = folds x epochs x N_train / measured_throughput, and require a measured throughput from a pilot of about 100 steps.
- GPU training uses Spot VMs. Every job checkpoints to the bucket every N steps, supports resume, syncs artifacts, then shuts the VM down.
- Stopped VMs still bill for disks. At the end of each phase list VMs and disks and propose deletion.
- Develop and debug on a tiny subsample locally before using a GPU.
- **Cloud Run Jobs fallback:** If Compute Engine GPU quota is unavailable or delayed, use
  `gcloud run jobs create ... --gpu=1 --no-gpu-zonal-redundancy --max-retries=3`.
  This auto-grants 3×L4 per region without a manual quota request. Same checkpoint-to-GCS
  and cost-safety rules apply. Show the full command and get confirmation before launching.
