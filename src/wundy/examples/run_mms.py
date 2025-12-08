"""
run_mms.py — Method of Manufactured Solutions (MMS) Verification
----------------------------------------------------------------

This script loads the MMS input model, runs the FE solver,
computes the exact analytical MMS solution, and reports error norms.

Usage:
    python -m wundy.examples.run_mms
"""

import numpy as np
import os
import wundy


def manufactured_solution(x):
    """Exact MMS displacement field u(x) = sin(pi x)."""
    return np.sin(np.pi * x)


def main():
    # Path to the MMS YAML file packaged inside the repository
    here = os.path.dirname(__file__)
    yaml_path = os.path.join(here, "mms_input.yaml")

    print("=====================================")
    print("   MMS VERIFICATION — WUNDY SOLVER   ")
    print("=====================================\n")
    print(f"Loading MMS model:\n  {yaml_path}\n")

    # -----------------------------------------
    # Load → preprocess → solve
    # -----------------------------------------
    data = wundy.ui.load(yaml_path)
    pre = wundy.ui.preprocess(data)
    result = wundy.solver.newton_solve_1d(pre)

    u = result["dofs"]
    x = pre["coords"][:, 0]
    u_exact = manufactured_solution(x)

    # -----------------------------------------
    # Compute errors
    # -----------------------------------------
    abs_error = np.abs(u - u_exact)
    max_error = np.max(abs_error)
    l2_error = np.sqrt(np.sum(abs_error**2) / len(abs_error))

    # -----------------------------------------
    # Print results
    # -----------------------------------------
    print("Exact MMS solution:     u(x) = sin(pi x)")
    print("Computed FE solution:   u_FE")
    print("\nNode :   x      u_FE         u_exact       |error|")
    print("-" * 55)

    for i in range(len(x)):
        print(f"{i:3d} : {x[i]:6.3f}   {u[i]:10.6f}   {u_exact[i]:10.6f}   {abs_error[i]:10.6f}")

    print("\n-------------------------------------")
    print(f"Max error   : {max_error:.3e}")
    print(f"L2 error    : {l2_error:.3e}")
    print("-------------------------------------\n")

    print("MMS verification complete.")
    print("If the solver is correct, errors should decrease with refinement.\n")


if __name__ == "__main__":
    main()
