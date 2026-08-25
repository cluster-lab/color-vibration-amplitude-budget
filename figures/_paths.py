# -*- coding: utf-8 -*-
"""Where everything lives, relative to this checkout, and a loader for the
condition table.

Every figure script imports from here rather than naming an absolute path, so
the repository runs wherever it is cloned.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
IMAGES = os.path.join(ROOT, "images")

# the three modules the figures import by name; see README for what each is
SHARE = os.path.join(ROOT, "code", "share")            # cfvi: cone coordinates
CODES = os.path.join(ROOT, "code", "codes")            # the CAM16+HK solver
MN = os.path.join(ROOT, "code", "chromagazer")         # MacAdam ellipse geometry

for p in (SHARE, MN):
    if p not in sys.path:
        sys.path.insert(0, p)
os.makedirs(OUT, exist_ok=True)


def conditions(block=None):
    """The condition table, optionally one system of it.

    Each entry carries the two linear display values of the condition, the
    trial counts pooled over observers, and, for system A, a cluster-bootstrap
    interval. Nothing below the condition is recorded; see README.
    """
    doc = json.load(open(os.path.join(DATA, "conditions.json"), encoding="utf-8"))
    rows = doc["conditions"]
    return rows if block is None else [r for r in rows if r["block"] == block]


def load_json(name):
    return json.load(open(os.path.join(DATA, name), encoding="utf-8"))
