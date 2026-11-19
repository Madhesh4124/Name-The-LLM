---
description: Launch one training experiment on a Spot GPU VM (needs explicit confirmation)
---
1. Ask for the experiment config name. Read configs/<exp>.yaml and the gcp-gpu-job skill.
2. Confirm a pilot of about 100 steps exists and report measured samples/s. If not, run the pilot first.
3. Show GPU-hours and the cost formula inputs. Ask the user to confirm the budget cap. STOP until confirmed.
4. Print the exact gcloud create command (from the skill) and wait for explicit confirmation. Then run it.
5. Copy the repo and configs to the VM, start the job inside tmux via scripts/run_job.sh (checkpoint, sync, shutdown).
6. Tell the user how to monitor: gcloud compute ssh, tmux attach, and gcloud storage ls on ckpt/ and oof/.
7. When artifacts appear in the bucket, run validate_oof.py on them and report metrics.
