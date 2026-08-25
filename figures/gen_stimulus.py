# -*- coding: utf-8 -*-
"""fig_stimulus: what the observer saw, from the experiment's own screenshots.

Three frames of a trial: the fixation cross that opens it, and the 2 deg patch
at the centre of the field for an achromatic and for a chromatic stimulus. A
screenshot catches one frame, so each patch is one of the two colors of its
pair, not the pair itself.

The captures are 3840x2160 and the mark sits at (1989, 839) in all three, so
each panel is cropped to the same window about that point; the patch would be
too small to read at the full field.
"""
import os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from figstyle import W, FS, FS_S, finish

from _paths import IMAGES as SRC
PANELS = [("092b96dce63848518c4114404cd060c7.JPG", "fixation cross"),
          ("9c36f85d7ae44896a3a37211cac06c10.JPG", "achromatic patch"),
          ("cc47ce1c5c4743dd9f9d4876e35c7adb.JPG", "chromatic patch")]
CX, CY = 1989, 839          # where the mark sits, measured on all three frames
CW, CH = 800, 600           # the crop, 4:3


def load(name):
    im = Image.open(os.path.join(SRC, name)).convert("RGB")
    im = im.crop((CX - CW // 2, CY - CH // 2, CX + CW // 2, CY + CH // 2))
    return np.asarray(im)


fig, axes = plt.subplots(1, 3, figsize=(W, W * 0.285),
                         gridspec_kw={"wspace": 0.05, "left": 0.005,
                                      "right": 0.995, "top": 0.90,
                                      "bottom": 0.02})
for ax, (name, lab) in zip(axes, PANELS):
    ax.imshow(load(name), interpolation="antialiased")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5); s.set_color("0.55")
    ax.set_title(lab, fontsize=FS_S, color="0.15", pad=2.5)
    print("  %-40s crop %dx%d" % (name[:12], CW, CH))

fig.canvas.draw()
for ax, tag in zip(axes, ("(a)", "(b)", "(c)")):
    fig.text(max(0.004, ax.get_position().x0), 0.955, tag, fontsize=FS,
             va="bottom")

finish(fig, "fig_stimulus")
