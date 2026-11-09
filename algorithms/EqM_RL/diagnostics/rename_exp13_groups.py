"""Rename the wandb groups of Experiment 13 so they say the algorithm is Aware and only the critic differs:
Exp13_awac_<env> -> Exp13_Aware-awacQ_<env>, Exp13_iql_<env> -> Exp13_Aware-iqlQ_<env>.
Only the group label in wandb changes (training is unaffected). summarize_exp13.py reads the group from each run's
command line in its local wandb files, so it keeps working. Runs that start later get the old name from the sweep
plan; run this again once all have started.

    python diagnostics/rename_exp13_groups.py            # rename
    python diagnostics/rename_exp13_groups.py --dry-run  # only list
"""

import argparse
import re

import wandb

ap = argparse.ArgumentParser()
ap.add_argument("--path", default="duck13/E", help="wandb entity/project")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

api = wandb.Api(timeout=60)
runs = api.runs(args.path, filters={"group": {"$regex": "^Exp13_(awac|iql)_"}})
n = 0
for r in runs:
    new = re.sub(r"^Exp13_(awac|iql)_", lambda m: f"Exp13_Aware-{m.group(1)}Q_", r.group)
    print(f"{r.name}  [{r.state}]  {r.group} -> {new}")
    if not args.dry_run:
        r.group = new
        r.update()
    n += 1
print(f"{'would rename' if args.dry_run else 'renamed'} {n} runs")
