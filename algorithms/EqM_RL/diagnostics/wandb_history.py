"""Read an Aware run's logged history from its local wandb file, without the network.

Aware names its checkpoint folder <name>-<env>-<run_id[:8]> and its wandb folder
run-<date-time>-<run_id> (in Aware/wandb/), so one maps to the other by the run id.
"""

import glob
import json
from pathlib import Path

import numpy as np

WEIGHT_KEYS = ("weighting/weights_mean", "weighting/weights_var", "weighting/weights_clip_frac",
               "weighting/adv_mean", "weighting/adv_std")


def wandb_dir_of(checkpoint_dir, wandb_root) -> str | None:
    """The wandb run folder of an Aware checkpoint folder, or None."""
    run_id8 = Path(checkpoint_dir).name.rsplit("-", 1)[-1]
    hits = sorted(glob.glob(f"{wandb_root}/run-*-{run_id8}-*"))
    return hits[-1] if hits else None


def read_aware_history(run_dir: str) -> dict:
    """eval: (n, 2) array of (step, eval/normalized_score); weights: (m, 6) array of
    (step, *WEIGHT_KEYS), one row per logged training step."""
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore

    ds = datastore.DataStore()
    ds.open_for_scan(glob.glob(run_dir + "/*.wandb")[0])
    ev, wt = [], []
    while (raw := ds.scan_data()) is not None:
        rec = pb.Record()
        rec.ParseFromString(raw)
        if rec.WhichOneof("record_type") != "history":
            continue
        h = {it.key or "/".join(it.nested_key): it.value_json for it in rec.history.item}
        if "_step" not in h:
            continue
        step = int(json.loads(h["_step"]))
        if "eval/normalized_score" in h:
            ev.append((step, float(json.loads(h["eval/normalized_score"]))))
        if WEIGHT_KEYS[0] in h:
            wt.append([step] + [float(json.loads(h[k])) for k in WEIGHT_KEYS])
    return dict(eval=np.array(ev), weights=np.array(wt))
