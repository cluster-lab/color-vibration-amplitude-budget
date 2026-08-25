# -*- coding: utf-8 -*-
"""Regenerate every figure of the paper into out/.

Run from anywhere:  python make_figures.py
Add --loci to re-solve the stimulus construction first; that step takes a few
minutes and overwrites data/fine_loci.json with the same values.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "figures")

# fine_loci.py comes first when asked for: gen_xy3.py reads what it writes
SOLVE = ["fine_loci.py"]
DRAW = [
    "gen_fig1g.py",      # Fig. 1  fig_concept
    "gen_xy3.py",        # Fig. 2  fig_xy
    "gen_design3.py",    # Fig. 3  fig_design
    "gen_stimulus.py",   # Fig. 4  fig_stimulus
    "gen_rq1_pred.py",   # Fig. 5  fig_rq1     and Fig. 9  fig_pred
    "gen_grids.py",      # Fig. 6  fig_rq2     and Fig. 8  fig_rq4
    "gen_axes.py",       # Fig. 7  fig_axes
    "gen_budget.py",     # Fig. 10 fig_budget
]


def run(script):
    print("\n=== %s " % script + "=" * (58 - len(script)))
    r = subprocess.run([sys.executable, script], cwd=FIG)
    if r.returncode:
        raise SystemExit("%s failed with exit code %d" % (script, r.returncode))


def main():
    scripts = (SOLVE if "--loci" in sys.argv else []) + DRAW
    for s in scripts:
        run(s)
    out = os.path.join(ROOT, "out")
    pdfs = sorted(f for f in os.listdir(out) if f.endswith(".pdf"))
    print("\n%d figures written to %s" % (len(pdfs), out))
    for f in pdfs:
        print("   %s" % f)


if __name__ == "__main__":
    main()
