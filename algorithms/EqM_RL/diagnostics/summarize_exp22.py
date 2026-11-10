"""Summarize Experiment 22 from one sweep folder, including incomplete runs.

Usage: python diagnostics/summarize_exp22.py sweep_folder
The discount is read from runs.csv because RESULT lines do not contain gamma.
"""
import argparse
import csv
import json
from pathlib import Path
import shlex
from statistics import mean, pstdev


SETTINGS = ((0.999, 0.9), (0.999, 0.8), (0.995, 0.9), (0.995, 0.8))
FINAL_STEPS = (800000, 900000, 1000000)
ENV_NAME = 'antsoccer-arena-navigate-singletask-v0'


def read_results(folder):
    results = {}
    with (folder / 'runs.csv').open(newline='') as handle:
        runs = list(csv.DictReader(handle))
    for run in runs:
        config = dict(item.split('=', 1) for item in shlex.split(run['overrides']))
        gamma = float(config['gamma'])
        expectile = float(config['expectile'])
        seed = int(config['seed'])
        assert config['env_name'] == ENV_NAME, config
        assert (gamma, expectile) in SETTINGS and seed in (1, 2, 3), config
        log = Path(run['log'])
        if not log.is_absolute():
            log = folder / log.parent.name / log.name
        if not log.exists():
            continue
        for line in log.read_text(errors='replace').splitlines():
            if not line.startswith('RESULT '):
                continue
            try:
                row = json.loads(line[7:])
            except json.JSONDecodeError:
                continue  # A running process may still be writing this line.
            assert row['env_name'] == ENV_NAME and row['seed'] == seed, row
            assert row['expectile'] == expectile, row
            if row['actor'] == 'trained':
                results[(gamma, expectile, seed, row['step'])] = row
    return results


def summarize(folder):
    results = read_results(folder)
    print('# Experiment 22: IEQM on AntSoccer arena\n')
    print(f'Source: `{folder}`. Default task 4; best of 32 EqM actions at K = 20; 50 episodes per evaluation.\n')
    print('Score = mean success (%) at 800k, 900k and 1M for each seed. ± is the population standard deviation across all three seed scores. Incomplete runs have no final score.\n')
    print('| Discount | Expectile | Seed | Last evaluation | Last success (%) | Final score (%) |')
    print('|---|---|---|---|---|---|')
    for gamma, expectile in SETTINGS:
        scores = []
        for seed in (1, 2, 3):
            rows = {t: r for (g, e, s, t), r in results.items() if (g, e, s) == (gamma, expectile, seed)}
            score = mean(rows[t]['success'] for t in FINAL_STEPS) if all(t in rows for t in FINAL_STEPS) else None
            if score is not None:
                scores.append(score)
            last = max(rows) if rows else None
            step = str(last) if last is not None else 'pending'
            success = f"{rows[last]['success']:.1f}" if last is not None else '–'
            final = f'{score:.1f}' if score is not None else '–'
            print(f'| {gamma} | {expectile} | {seed} | {step} | {success} | {final} |')
        aggregate = f'{mean(scores):.1f} ± {pstdev(scores):.1f}' if len(scores) == 3 else 'pending'
        print(f'| {gamma} | {expectile} | **all 3 seeds** | | | **{aggregate}** |')
    for gamma, expectile in SETTINGS:
        print(f'\n## Success over training: discount {gamma}, expectile {expectile}\n')
        print('| Step | Seed 1 | Seed 2 | Seed 3 |')
        print('|---|---|---|---|')
        steps = sorted({t for g, e, s, t in results if (g, e) == (gamma, expectile)})
        for step in steps:
            values = [f"{results[(gamma, expectile, seed, step)]['success']:.1f}" if (gamma, expectile, seed, step) in results else '–' for seed in (1, 2, 3)]
            print(f'| {step} | ' + ' | '.join(values) + ' |')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    summarize(parser.parse_args().folder)
