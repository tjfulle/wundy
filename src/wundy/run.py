import argparse
import sys

import wundy
from wundy import first
from wundy import ui

def run():
    p = argparse.ArgumentParser()
    p.add_argument("file", help = "Wundy yaml file")
    args = p.parse_args()
    with open(args.file) as fh:
        data = ui.load(fh)
    inp = ui.preprocess(data)
    soln = wundy.first.first_fe_code(
        inp["coords"],
        inp["blocks"],
        inp["bcs"],
        inp["dload"],
        inp["materials"],
        inp["block_elem_map"],
    )

    # Present a concise summary of the solution
    print("== Wundy solution ==")
    print(f"converged: {soln.get('converged', False)}")
    print(f"iterations: {soln.get('num_iter')}")
    dofs = soln.get("dofs")
    if dofs is None:
        dofs = soln.get("u")
    if dofs is not None:
        print("dofs:")
        print(dofs)
    residual = soln.get("residual_norm")
    if residual is not None:
        print(f"residual_norm: {residual}")
    force = soln.get("force")
    if force is not None:
        print("nodal_force:")
        print(force)
