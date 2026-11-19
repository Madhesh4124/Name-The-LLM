---
description: Phase 0 data audit; writes docs/eda_report.md
---
1. Read configs/labels.json (create from train labels if missing) and load train.csv and test.csv.
// turbo
2. Compute: class counts, passage length (words and chars) per class, duplicate texts, missing values.
3. Check and report each of these explicitly:
   - Does train.csv contain a prompt id or any grouping metadata?
   - Do markdown, newlines, curly quotes and em/en dashes survive in the text? Counts per class.
   - Are passages truncated (do endings correlate with class)?
   - Does test appear to contain several answers to the same prompt? Evidence either way.
   - Does sklearn's char analyzer collapse newlines in this data? Show a 2-line test.
4. Write docs/eda_report.md with findings and the implications for the plan. Do not model anything.
