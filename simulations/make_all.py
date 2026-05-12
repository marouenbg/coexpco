"""
Run all three experiments and write figures into manuscript/figs/.
"""

import sys
import pathlib
import time

THIS = pathlib.Path(__file__).parent
sys.path.insert(0, str(THIS))

from exp1_loo_correlation import run as run_exp1
from exp2_lioness_application import run as run_exp2
from exp3_two_isn import run as run_exp3
from exp4_timing import run as run_exp4


def main():
    t0 = time.time()
    print("=" * 72)
    print("EXP 1 -- benchmark on C_{alpha-q}")
    print("=" * 72)
    out1 = run_exp1()

    print("\n" + "=" * 72)
    print("EXP 2 -- propagation into LIONESS edges")
    print("=" * 72)
    out2 = run_exp2()

    print("\n" + "=" * 72)
    print("EXP 3 -- two-cluster signal vs method noise")
    print("=" * 72)
    out3 = run_exp3()

    print("\n" + "=" * 72)
    print("EXP 4 -- wall-time speedup of M1 over brute-force")
    print("=" * 72)
    out4 = run_exp4()

    dt = time.time() - t0
    print("\n" + "=" * 72)
    print(f"DONE in {dt:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
