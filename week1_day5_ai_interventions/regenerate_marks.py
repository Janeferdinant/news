#!/usr/bin/env python3
"""Regenerate the SYNTHETIC marks used by this folder.

Every pre-test and post-test score, every group assignment and every student id in this folder is produced here
from a seeded random number generator. None of it describes real students. Running the script again with the same
seed gives byte-identical files.

It rewrites three things so they always agree with each other:
  * intervention_data.csv         student_id, group, pre_score, post_score
  * assets/plot.js                the gains drawn by the interactive chart on report.html
  * report.html                   the one-sentence summary between the "synthetic-summary" markers

Usage (Python 3.8+, standard library only; run from anywhere):
    python week1_day5_ai_interventions/regenerate_marks.py                 # default: seed 42, 10 students per group
    python week1_day5_ai_interventions/regenerate_marks.py --seed 7 --per-group 15
"""
import argparse
import csv
import json
import random
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Assumed average gains of the two groups in the synthetic world, and the spread of everything else.
GROUPS = (("control", 2.0), ("ai", 6.0))
PRE_MEAN, PRE_SD = 70.0, 6.0
GAIN_SD = 3.0


def clamp(value, low, high):
    return max(low, min(high, value))


def make_rows(seed, per_group):
    """Return a list of (student_id, group, pre_score, post_score) with integer scores in 0..100."""
    rng = random.Random(seed)
    rows = []
    student_id = 1
    for group, mean_gain in GROUPS:
        for _ in range(per_group):
            pre = clamp(round(rng.gauss(PRE_MEAN, PRE_SD)), 40, 95)
            gain = round(rng.gauss(mean_gain, GAIN_SD))
            post = clamp(pre + gain, 0, 100)
            rows.append((student_id, group, pre, post))
            student_id += 1
    return rows


def gains_by_group(rows):
    out = {group: [] for group, _ in GROUPS}
    for _, group, pre, post in rows:
        out[group].append(post - pre)
    return out


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["student_id", "group", "pre_score", "post_score"])
        writer.writerows(rows)


def write_plot_js(gains, path):
    names = {"control": "Control", "ai": "AI"}
    traces = ",\n".join(
        f"    {{ y: {json.dumps(gains[group])}, type: 'box', name: '{names[group]}' }}" for group, _ in GROUPS
    )
    js = (
        "// Synthetic data, written by ../regenerate_marks.py. Do not edit by hand.\n"
        "document.addEventListener('DOMContentLoaded', function () {\n"
        "  var data = [\n"
        f"{traces}\n"
        "  ];\n"
        "  var layout = {\n"
        "    title: 'Learning Gains by Group (synthetic data)',\n"
        "    yaxis: { title: 'Score Gain' }\n"
        "  };\n"
        "  Plotly.newPlot('interactive-plot', data, layout);\n"
        "});\n"
    )
    path.write_text(js, encoding="utf-8", newline="\n")


def summary_sentence(gains, seed):
    control = statistics.mean(gains["control"])
    ai = statistics.mean(gains["ai"])
    if ai > control:
        direction = "greater"
    elif ai < control:
        direction = "smaller"
    else:
        direction = "the same"
    n = len(gains["control"])
    return (
        f"In this synthetic sample (seed {seed}, {n} per group), the AI group's mean gain was {ai:.1f} points "
        f"and the control group's was {control:.1f}, so the AI group's gain was {direction}. "
        "This only reflects the values the script was told to generate."
    )


def update_report(gains, seed, path):
    html = path.read_text(encoding="utf-8")
    pattern = re.compile(r"(<!-- synthetic-summary:start -->).*?(<!-- synthetic-summary:end -->)", re.S)
    if not pattern.search(html):
        raise SystemExit("report.html has no synthetic-summary markers")
    html = pattern.sub(lambda m: f"{m.group(1)}{summary_sentence(gains, seed)}{m.group(2)}", html)
    path.write_text(html, encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    parser.add_argument("--per-group", type=int, default=10, help="students per group (default 10)")
    args = parser.parse_args()
    if args.per_group < 2:
        parser.error("--per-group must be at least 2")

    rows = make_rows(args.seed, args.per_group)
    gains = gains_by_group(rows)
    write_csv(rows, HERE / "intervention_data.csv")
    write_plot_js(gains, HERE / "assets" / "plot.js")
    update_report(gains, args.seed, HERE / "report.html")

    print(f"seed={args.seed} per_group={args.per_group} rows={len(rows)}")
    for group, _ in GROUPS:
        g = gains[group]
        print(f"{group}: n={len(g)} mean_gain={statistics.mean(g):.2f} sd={statistics.stdev(g):.2f}")


if __name__ == "__main__":
    main()
