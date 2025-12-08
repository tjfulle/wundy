#!/usr/bin/env python3
"""
WUNDY Command-Line Runner

Usage:
    python bin/run.py <input.yaml>

This script loads a YAML file, preprocesses it using the WUNDY
input schema, then solves the problem using the nonlinear
Newton–Raphson solver from wundy.solver (wrapped by wundy.solve).
"""

import sys
import numpy as np

import wundy


def main():
    if len(sys.argv) < 2:
        print("Usage: python bin/run.py <input.yaml>")
        sys.exit(1)

    fname = sys.argv[1]
    print(f"\n=== WUNDY: Solving {fname} ===\n")

    # High-level solver call
    result = wundy.solve(fname)

    # ------------------------------------------------------------
    # Extract results
    # ------------------------------------------------------------
    u = np.asarray(result.get("dofs", result.get("U", [])))
    R = np.asarray(result.get("reactions", []))
    fint = np.asarray(result.get("internal_forces", []))
    conv = result.get("convergence", [])

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------
    print("Displacement vector (u):")
    print(u)

    if R.size > 0:
        print("\nReaction forces (R):")
        print(R)

    if fint.size > 0:
        print("\nInternal element forces:")
        print(fint)

    # Newton convergence history
    if conv:
        print("\nNewton Iteration Log:")
        for it, res in conv:
            print(f"  Iter {it:2d} : residual = {res:.6e}")

    print("\n=== WUNDY SOLVER COMPLETE ===\n")


if __name__ == "__main__":
    main()
