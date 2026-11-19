---
description: Read-only GCP readiness check (auth, quota, zone, bucket)
---
1. Read configs/gcp.env.
// turbo
2. Run: gcloud auth list ; gcloud config list. Confirm the active project equals PROJECT_ID.
// turbo
3. Run: gcloud compute regions describe $REGION --format="yaml(quotas)". Report NVIDIA_L4_GPUS and PREEMPTIBLE_NVIDIA_L4_GPUS limits and usage [verify metric names].
// turbo
4. Run: gcloud compute accelerator-types list --filter="zone:$ZONE". Report whether L4 exists in the zone; suggest alternatives if not.
// turbo
5. Run: gcloud storage ls $BUCKET.
6. Ask the user to confirm a billing budget with alerts exists. Do not create anything.
7. Output a pass/fail table and the next action for each failure.
