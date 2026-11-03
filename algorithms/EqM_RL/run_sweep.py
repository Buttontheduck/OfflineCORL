"""Run a hyperparameter sweep from a YAML file, many runs at once on one GPU.

Usage (from algorithms/EqM_RL/):
    python run_sweep.py sweeps/temperature_minari.yaml             # run the sweep
    python run_sweep.py sweeps/temperature_minari.yaml --dry-run   # only list the runs
    python run_sweep.py sweeps/temperature_minari.yaml --parallel 8

Any Python with PyYAML can start this file (the hawk, eagle or base env). Each training
run is started with the Python of the conda env its target needs: `aware` (D4RL) runs
in hawk, `aware_minari` and `awac_minari` run in eagle. The sweep file format is
described in sweeps/README.md.

Output: sweep_runs/<sweep name>_<date-time>/ with a copy of the sweep file, plan.txt
(every command), runs.csv (status of every run) and one folder per run with its log.
"""

import argparse
import csv
import datetime
import itertools
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

TARGETS = {
    # D4RL data, old gym + mujoco-py, needs the MuJoCo 2.1 binaries
    "aware": dict(script=HERE / "Aware" / "main.py", conda_env="hawk",
                  style="hydra", data="d4rl", name_prefix="Aware"),
    # Minari data, gymnasium + new mujoco bindings
    "aware_minari": dict(script=HERE / "Aware_Minari" / "main.py", conda_env="eagle",
                         style="hydra", data="minari", name_prefix="Aware"),
    # Gaussian AWAC baseline on Minari (pyrallis command line)
    "awac_minari": dict(script=HERE / "Awac_Minari.py", conda_env="eagle",
                        style="pyrallis", data="minari", name_prefix="AWAC"),
    # CORL's Gaussian AWAC on D4RL (algorithms/offline/awac.py, pyrallis command line)
    "awac_d4rl": dict(script=HERE.parent / "offline" / "awac.py", conda_env="hawk",
                      style="pyrallis", data="d4rl", name_prefix="AWAC"),
}
SWEEP_KEYS = {"target", "parallel", "stagger", "conda_env", "fixed", "grid"}
MUJOCO_BIN = Path.home() / ".mujoco" / "mujoco210" / "bin"


# ----------------------------------------------------------------------------- sweep file

def load_sweep(path: Path) -> dict:
    with open(path) as f:
        spec = yaml.safe_load(f) or {}
    unknown = set(spec) - SWEEP_KEYS
    if unknown:
        sys.exit(f"{path}: unknown keys {sorted(unknown)}; allowed: {sorted(SWEEP_KEYS)}")
    if spec.get("target") not in TARGETS:
        sys.exit(f"{path}: 'target' must be one of {list(TARGETS)}, got {spec.get('target')!r}")
    fixed = spec.get("fixed") or {}
    if not isinstance(fixed, dict):
        sys.exit(f"{path}: 'fixed' must be a mapping of key: value")
    spec["fixed"] = fixed
    return spec


def expand_grid(grid) -> list:
    """Cartesian product of every grid. `grid` is one mapping or a list of mappings;
    a list of mappings gives the union of their products (duplicates removed)."""
    if not grid:
        return [{}]
    grids = grid if isinstance(grid, list) else [grid]
    combos, seen = [], set()
    for g in grids:
        if not isinstance(g, dict):
            sys.exit("'grid' must be a mapping, or a list of mappings")
        keys = list(g)
        values = [v if isinstance(v, list) else [v] for v in g.values()]
        for combo in itertools.product(*values):
            run = dict(zip(keys, combo))
            key = repr(sorted(run.items(), key=lambda kv: kv[0]))
            if key not in seen:
                seen.add(key)
                combos.append(run)
    return combos


def tag_of(combo: dict) -> str:
    """Short readable id of one grid point, e.g. temperature0.3_num_step10_seed1.
    Values with '/' (dataset ids) keep only their last part: simple-v0."""
    parts = []
    for k, v in combo.items():
        val = re.sub(r"[^\w.\-]+", "", str(v).rsplit("/", 1)[-1])
        parts.append(f"{k.split('.')[-1]}{val}")
    return "_".join(parts) or "base"


# ----------------------------------------------------------------------------- command line

_SAFE = re.compile(r"[A-Za-z0-9_./\-]+")


def hydra_value(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(hydra_value(x) for x in v) + "]"
    if isinstance(v, dict):
        sys.exit(f"Mapping values are not supported, use dotted keys instead: {v}")
    s = str(v)
    if _SAFE.fullmatch(s):
        return s
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def pyrallis_value(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def build_cmd(python: str, target: dict, overrides: dict, run_dir: Path) -> list:
    cmd = [python, str(target["script"])]
    if target["style"] == "hydra":
        cmd += [f"{k}={hydra_value(v)}" for k, v in overrides.items()]
        cmd.append(f"hydra.run.dir={hydra_value(str(run_dir / 'hydra'))}")
    else:
        cmd += [f"--{k}={pyrallis_value(v)}" for k, v in overrides.items()]
    return cmd


# ----------------------------------------------------------------------------- environment

def conda_python(env_name: str) -> str:
    candidates = []
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix and Path(prefix).name == env_name:
        candidates.append(Path(prefix) / "bin" / "python")
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda:
        base = Path(conda).resolve().parent.parent
        candidates += [base / "envs" / env_name / "bin" / "python",
                       base / "bin" / "python" if env_name == "base" else None]
    for c in candidates:
        if c is not None and c.exists():
            return str(c)
    sys.exit(f"Cannot find the Python of conda env '{env_name}'. "
             f"Set 'conda_env:' in the sweep file to an existing env.")


def child_environment(target: dict) -> dict:
    env = os.environ.copy()
    # Each run is GPU-bound with tiny CPU work; without this, every process starts one
    # thread per CPU core and 16 runs fight over the CPU.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env["PYTHONUNBUFFERED"] = "1"
    if target["data"] == "d4rl":
        # mujoco-py needs the MuJoCo 2.1 binaries on the library path. Starting the env's
        # Python directly skips `conda activate`, so conda env vars would not apply here.
        env["LD_LIBRARY_PATH"] = f"{MUJOCO_BIN}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
        env.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    return env


def default_env_name(target: dict) -> str:
    if target["style"] == "hydra":
        with open(target["script"].parent / "config.yaml") as f:
            return yaml.safe_load(f)["env_name"]
    m = re.search(r'env_name:\s*str\s*=\s*"([^"]+)"', target["script"].read_text())
    if not m:
        sys.exit(f"Cannot find the default env_name in {target['script']}")
    return m.group(1)


def prefetch_datasets(python: str, target: dict, env: dict, names: list) -> None:
    """Download each dataset once, before the runs start, so parallel runs do not all
    try to download the same file at the same time."""
    if target["data"] == "d4rl":
        code = "import sys, gym, d4rl\ngym.make(sys.argv[1]).get_dataset()"
    else:
        code = ("import sys, minari\n"
                "if sys.argv[1] not in minari.list_local_datasets():\n"
                "    minari.download_dataset(sys.argv[1])")
    for name in names:
        print(f"Checking dataset {name} ...", flush=True)
        r = subprocess.run([python, "-c", code, name], env=env, cwd=target["script"].parent,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:], sep="\n")
            sys.exit(f"Could not load dataset {name}")


def mps_running():
    """True/False if MPS is running, None if it cannot be checked."""
    if shutil.which("pgrep") is None:
        return None
    # process names are cut to 15 characters: 'nvidia-cuda-mps' matches control and server
    return subprocess.run(["pgrep", "-x", "nvidia-cuda-mps"], capture_output=True).returncode == 0


# ----------------------------------------------------------------------------- runner

def now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep", type=Path, help="sweep YAML file")
    ap.add_argument("--parallel", type=int, help="runs at once (default: from the file, else 16)")
    ap.add_argument("--dry-run", action="store_true", help="only print the runs and commands")
    ap.add_argument("--out", type=Path, default=HERE / "sweep_runs", help="output root folder")
    args = ap.parse_args()

    spec = load_sweep(args.sweep)
    target = TARGETS[spec["target"]]
    parallel = args.parallel or spec.get("parallel") or 16
    stagger = float(spec.get("stagger", 2.0))
    python = conda_python(spec.get("conda_env") or target["conda_env"])
    env = child_environment(target)

    sweep_name = args.sweep.stem
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sweep_dir = (args.out / f"{sweep_name}_{stamp}").resolve()

    fixed = dict(spec["fixed"])
    fixed.setdefault("group", sweep_name)
    name_prefix = fixed.pop("name", None) or target["name_prefix"]

    runs = []
    for i, combo in enumerate(expand_grid(spec.get("grid"))):
        tag = tag_of(combo)
        overrides = {**fixed, **combo, "name": f"{name_prefix}_{tag}"}
        run_dir = sweep_dir / f"{i:03d}_{tag}"
        runs.append(dict(idx=i, tag=tag, overrides=overrides, run_dir=run_dir,
                         cmd=build_cmd(python, target, overrides, run_dir),
                         status="waiting", returncode="", minutes=""))

    print(f"Sweep {sweep_name}: {len(runs)} runs of '{spec['target']}', {parallel} at once")
    print(f"Python: {python}")
    if args.dry_run:
        for r in runs:
            print(f"\n[{r['idx']:03d}] {r['tag']}\n  " + shlex.join(r["cmd"]))
        return 0

    if parallel > 1 and mps_running() is False:
        print("WARNING: CUDA MPS is not running, so parallel runs take turns on the GPU "
              "(about 5x one run in total instead of 16x). Start it with: nvidia-cuda-mps-control -d")

    env_names = sorted({str(r["overrides"].get("env_name", default_env_name(target))) for r in runs})
    prefetch_datasets(python, target, env, env_names)

    sweep_dir.mkdir(parents=True)
    shutil.copy(args.sweep, sweep_dir / "sweep.yaml")
    with open(sweep_dir / "plan.txt", "w") as f:
        for r in runs:
            f.write(f"[{r['idx']:03d}] {r['tag']}\n{shlex.join(r['cmd'])}\n\n")
    print(f"Output: {sweep_dir}\n")

    def write_csv():
        with open(sweep_dir / "runs.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "tag", "status", "exit_code", "minutes", "log", "overrides"])
            for r in runs:
                w.writerow([r["idx"], r["tag"], r["status"], r["returncode"], r["minutes"],
                            r["run_dir"] / "stdout.log",
                            " ".join(f"{k}={v}" for k, v in r["overrides"].items())])

    def on_sigterm(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_sigterm)

    waiting, running = deque(runs), {}
    write_csv()
    try:
        while waiting or running:
            changed = False
            for proc, r in list(running.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                r["log_file"].close()
                r["returncode"] = rc
                r["minutes"] = f"{(time.time() - r['t0']) / 60:.1f}"
                r["status"] = "done" if rc == 0 else "failed"
                del running[proc]
                changed = True
                if rc == 0:
                    print(f"[{now()}] done   {r['idx']:03d} {r['tag']} ({r['minutes']} min)")
                else:
                    print(f"[{now()}] FAILED {r['idx']:03d} {r['tag']} (exit {rc}), "
                          f"see {r['run_dir'] / 'stdout.log'}")
            while waiting and len(running) < parallel:
                r = waiting.popleft()
                r["run_dir"].mkdir(parents=True, exist_ok=True)
                r["log_file"] = open(r["run_dir"] / "stdout.log", "w")
                # own session: Ctrl-C in the terminal reaches only this runner, which then
                # stops the runs itself
                proc = subprocess.Popen(r["cmd"], stdout=r["log_file"], stderr=subprocess.STDOUT,
                                        cwd=target["script"].parent, env=env, start_new_session=True)
                r["t0"], r["status"] = time.time(), "running"
                running[proc] = r
                changed = True
                print(f"[{now()}] start  {r['idx']:03d} {r['tag']}")
                if waiting and len(running) < parallel:
                    time.sleep(stagger)  # spread out the start-up (data loading) a little
            if changed:
                n_done = sum(r["status"] == "done" for r in runs)
                n_fail = sum(r["status"] == "failed" for r in runs)
                print(f"[{now()}] finished {n_done + n_fail}/{len(runs)} (failed {n_fail}) | "
                      f"running {len(running)} | waiting {len(waiting)}", flush=True)
                write_csv()
            time.sleep(2)
    except KeyboardInterrupt:
        print(f"\n[{now()}] Stopping {len(running)} running runs (Ctrl-C again is ignored) ...")
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        for proc in running:
            proc.send_signal(signal.SIGINT)  # lets each run close its W&B run
        deadline = time.time() + 60
        for proc, r in running.items():
            try:
                proc.wait(timeout=max(1.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            r["log_file"].close()
            r["status"], r["returncode"] = "stopped", proc.returncode
            r["minutes"] = f"{(time.time() - r['t0']) / 60:.1f}"
        for r in runs:
            if r["status"] == "waiting":
                r["status"] = "not started"
        write_csv()
        print(f"Stopped. Status: {sweep_dir / 'runs.csv'}")
        return 130

    n_fail = sum(r["status"] == "failed" for r in runs)
    print(f"\nAll {len(runs)} runs finished, {n_fail} failed. Status: {sweep_dir / 'runs.csv'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
