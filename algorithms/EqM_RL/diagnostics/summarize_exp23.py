"""Summarize Experiment 23's D4RL normalized returns from one sweep's local logs.

Usage: python diagnostics/summarize_exp23.py sweep_folder
Only seeds with evaluations at 800k, 900k and 1M receive a final score.
"""
import argparse
import csv
import json
from pathlib import Path
import shlex
from statistics import mean, pstdev


ENVS = ('halfcheetah-medium-v2', 'halfcheetah-medium-replay-v2', 'halfcheetah-medium-expert-v2')
EXPECTILES = (0.8, 0.9)
FINAL_STEPS = (800000, 900000, 1000000)


def read_results(folder):
    results = {}
    with (folder / 'runs.csv').open(newline='') as handle:
        runs = list(csv.DictReader(handle))
    for run in runs:
        config = dict(item.split('=', 1) for item in shlex.split(run['overrides']))
        env, expectile, seed = config['env_name'], float(config['expectile']), int(config['seed'])
        assert env in ENVS and expectile in EXPECTILES and seed in (1, 2, 3), config
        assert float(config['gamma']) == 0.99, config
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
                continue
            assert (row['env_name'], row['expectile'], row['seed'], row['gamma']) == (env, expectile, seed, 0.99), row
            if row['actor'] == 'trained':
                results[env, expectile, seed, row['step']] = row
    return results


def summarize(folder):
    results = read_results(folder)
    print('# Experiment 23: IEQM on D4RL HalfCheetah\n')
    print(f'Source: `{folder}`. Discount 0.99; best of 32 EqM actions, K = 20; 50 episodes per evaluation.\n')
    print('Scores are D4RL normalized returns, not success percentages. Each seed score averages 800k, 900k and 1M. ± is population standard deviation across all three seeds; unfinished settings remain pending.\n')
    print('| Dataset | Expectile | Seed 1 | Seed 2 | Seed 3 | Mean ± std |')
    print('|---|---|---|---|---|---|')
    for env in ENVS:
        for e in EXPECTILES:
            scores = [mean(results[env, e, s, t]['normalized_score'] for t in FINAL_STEPS)
                      if all((env, e, s, t) in results for t in FINAL_STEPS) else None for s in (1, 2, 3)]
            aggregate = f'{mean(scores):.1f} ± {pstdev(scores):.1f}' if all(v is not None for v in scores) else 'pending'
            values = ' | '.join('pending' if v is None else f'{v:.1f}' for v in scores)
            print(f'| {env} | {e} | {values} | **{aggregate}** |')
    print('\n## Latest evaluation per run\n')
    print('| Dataset | Expectile | Seed | Step | Normalized return | Raw return | Flip rate |')
    print('|---|---|---|---|---|---|---|')
    for env in ENVS:
        for e in EXPECTILES:
            for s in (1, 2, 3):
                rows = {t: row for (name, ex, seed, t), row in results.items() if (name, ex, seed) == (env, e, s)}
                last = max(rows) if rows else None
                if last is None:
                    print(f'| {env} | {e} | {s} | pending | – | – | – |')
                else:
                    r = rows[last]
                    print(f"| {env} | {e} | {s} | {last} | {r['normalized_score']:.1f} | {r['score']:.1f} | {r['flip_rate']:.3f} |")
    for env in ENVS:
        for e in EXPECTILES:
            print(f'\n## Learning curve: {env}, expectile {e}\n')
            print('| Step | Seed 1 | Seed 2 | Seed 3 |')
            print('|---|---|---|---|')
            for t in sorted({t for name, ex, s, t in results if (name, ex) == (env, e)}):
                values = [f"{results[env, e, s, t]['normalized_score']:.1f}" if (env, e, s, t) in results else '–'
                          for s in (1, 2, 3)]
                print(f'| {t} | ' + ' | '.join(values) + ' |')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    summarize(parser.parse_args().folder)
