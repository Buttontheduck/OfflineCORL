"""Read Experiment 21's local RESULT logs. No W&B connection needed.

Usage: python diagnostics/summarize_exp21.py [sweep_folder]
Final scores require all three evaluations at 800k, 900k and 1M.
"""
import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


def summarize(folder):
    results = {}
    for log in sorted(folder.glob('*/stdout.log')):
        for line in log.read_text(errors='replace').splitlines():
            if not line.startswith('RESULT '):
                continue
            try:
                row = json.loads(line[7:])
            except json.JSONDecodeError:
                continue  # a live log may end partway through a line
            if row['env_name'] != 'antmaze-large-navigate-singletask-v0' or row['actor'] != 'trained':
                continue
            results[(row['expectile'], row['seed'], row['step'])] = row
    print('# Experiment 21: IEQM on AntMaze large\n')
    print(f'Source: `{folder}`. Discount 0.99; best of 32 actions, K = 20; 50 episodes per evaluation.\n')
    print('Score = mean success at 800k, 900k and 1M within each seed. ± is the population standard deviation across seeds. Incomplete seeds have no final score.\n')
    print('| Expectile | Seed | Last evaluation | Last success (%) | Final score (%) |')
    print('|---|---|---|---|---|')
    for expectile in (0.9, 0.8):
        final_scores = []
        for seed in (1, 2, 3):
            rows = {step: row for (e, s, step), row in results.items() if e == expectile and s == seed}
            final_steps = (800000, 900000, 1000000)
            score = mean(rows[t]['success'] for t in final_steps) if all(t in rows for t in final_steps) else None
            if score is not None:
                final_scores.append(score)
            last = max(rows) if rows else None
            last_step = str(last) if last is not None else 'pending'
            last_success = f"{rows[last]['success']:.1f}" if last is not None else '–'
            final = f'{score:.1f}' if score is not None else '–'
            print(f'| {expectile} | {seed} | {last_step} | {last_success} | {final} |')
        aggregate = f'{mean(final_scores):.1f} ± {pstdev(final_scores):.1f}' if len(final_scores) == 3 else 'pending'
        print(f'| {expectile} | **all 3 seeds** | | | **{aggregate}** |')
    for expectile in (0.9, 0.8):
        print(f'\n## Success over training: expectile {expectile}\n')
        print('| Step | Seed 1 | Seed 2 | Seed 3 |')
        print('|---|---|---|---|')
        for step in sorted({t for e, s, t in results if e == expectile}):
            values = [f"{results[(expectile, s, step)]['success']:.1f}" if (expectile, s, step) in results else '–' for s in (1, 2, 3)]
            print(f'| {step} | ' + ' | '.join(values) + ' |')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', nargs='?', type=Path)
    args = parser.parse_args()
    folders = sorted((Path(__file__).resolve().parents[1] / 'sweep_runs').glob('exp21_ogbench_ieqm_antmaze_large_2*'))
    if args.folder is None and not folders:
        parser.error('No Experiment 21 sweep folder exists yet.')
    summarize(args.folder if args.folder is not None else folders[-1])
