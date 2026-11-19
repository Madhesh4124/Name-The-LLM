---
description: List and (with confirmation) delete project VMs and disks at the end of a phase
---
// turbo
1. Run: gcloud compute instances list ; gcloud compute disks list ; gcloud run jobs list --region="$REGION". Show a table with status and size.
2. Propose what to delete and why. Never delete anything under the bucket.
3. Ask for explicit confirmation, then delete only the confirmed resources.
// turbo
4. Re-list to verify nothing billable is left.
