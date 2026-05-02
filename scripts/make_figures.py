from __future__ import annotations

import os
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "generate_canonical_results.py")

print("Running:", GEN)
runpy.run_path(GEN, run_name="__main__")
