"""
Windower: Hierarchical Windowing and RDF Knowledge Graph Export
HPC-Safe Version for fry.cs.wright.edu / Singularity
"""

import argparse
from typing import Sequence, Optional, Tuple, List, Literal

import random
import os, sys

# ================ HPC SAFE PATH HANDLING ================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tools
# Base directory:
#   • On fry:     SLURM_SUBMIT_DIR  (directory you submitted job from)
#   • Locally:    directory of this file
BASE_DIR = os.environ.get(
    "SLURM_SUBMIT_DIR",
    os.path.dirname(os.path.abspath(__file__))
)

# KGs output folder:
#   • Tries $SCRATCH first (HPC high-speed workspace)
#   • Falls back to BASE_DIR/KGs
SCRATCH = os.environ.get("SCRATCH")
if SCRATCH:
    KG_DIR = os.path.join(SCRATCH, "KGs")
else:
    KG_DIR = os.path.join(BASE_DIR, "KGs")

os.makedirs(KG_DIR, exist_ok=True)

print(f"[INFO] Windower BASE_DIR: {BASE_DIR}")
print(f"[INFO] Writing KGs to:   {KG_DIR}")

# ========================================================


class Window:
    def __init__(self, elements: Sequence[Tuple[str, float]], n: int, path="root"):
        self.elements: List[Tuple[str, float]] = list(elements)
        self.n = n
        self.path = path
        self.left: Optional["Window"] = None
        self.right: Optional["Window"] = None

    def __repr__(self):
        return f"Window(path={self.path}, n={self.n}, size={len(self.elements)})"


def windower(pairs: Sequence[Tuple[str, float]], n: int = 1, path="root") -> Window:
    node = Window(pairs, n, path=path)
    if n <= 1 or len(pairs) <= 1:
        return node
    left, right = tools.split_pairs(pairs)
    node.left = windower(left, n - 1, path + "->L")
    node.right = windower(right, n - 1, path + "->R")
    return node


def print_windows(node: Window, ndigits=3):
    if node is None:
        return
    elems = [(e, round(v, ndigits)) for e, v in node.elements]
    print(f"Window path={node.path}, depth={node.n}, elems={elems}")
    if node.left:
        print_windows(node.left, ndigits)
    if node.right:
        print_windows(node.right, ndigits)


# ================ KG BUILDING (HPC SAFE) ================

def build_kg(root: Window, norm_pairs: List[Tuple[str, float]], outfile: str,
             lt_mode: Literal["sequential", "pairwise", "rand5", "pairwise5", "pairwise10"],
             reverse):

    # Force output into the HPC-safe KG_DIR
    out_path = os.path.join(KG_DIR, outfile)
    vals_path = f"{out_path}.val"

    print(f"[INFO] Writing KG to: {out_path}")
    print(f"[INFO] Writing values to: {vals_path}")

    with open(out_path, "w") as f, open(vals_path, "w") as vf:

        # --------- Write window triples ---------
        def add_window_triples(node: Window):
            window_label = f"Window_{node.path.replace('->', '_')}"

            for e, v in node.elements:
                f.write(f"{e}\tinWindow\t{window_label}\n")

            if node.left:
                add_window_triples(node.left)
            if node.right:
                add_window_triples(node.right)

        add_window_triples(root)

        # --------- Write lessThan edges ---------
        sorted_pairs = sorted(norm_pairs, key=lambda x: x[1], reverse=reverse)
        print(f"[DEBUG] First 10 sorted pairs: {sorted_pairs[:10]}")

        if lt_mode == "pairwise":
            for i, (e1, _) in enumerate(sorted_pairs):
                for e2, _ in sorted_pairs[i+1:]:
                    f.write(f"{e1}\tlessThan\t{e2}\n")

        elif lt_mode == "pairwise5":
            for i, (e1, _) in enumerate(sorted_pairs):
                for (e2, _) in sorted_pairs[i+1:i+6]:
                    f.write(f"{e1}\tlessThan\t{e2}\n")

        elif lt_mode == "pairwise10":
            for i, (e1, _) in enumerate(sorted_pairs):
                for (e2, _) in sorted_pairs[i+1:i+11]:
                    f.write(f"{e1}\tlessThan\t{e2}\n")

        elif lt_mode == "sequential":
            for (e1, _), (e2, _) in zip(sorted_pairs, sorted_pairs[1:]):
                f.write(f"{e1}\tlessThan\t{e2}\n")

        elif lt_mode == "rand5":
            for i, (e1, _) in enumerate(sorted_pairs):
                remain = sorted_pairs[i+1:]
                if not remain:
                    continue
                k = min(5, len(remain))
                picks = random.sample(remain, k)
                for (e2, _) in picks:
                    f.write(f"{e1}\tlessThan\t{e2}\n")

        # --------- Write values ---------
        for e, v in sorted_pairs:
            vf.write(f"{e}\t{v}\n")


# ================ PIPELINE ================

def run_windower_pipeline(
    entities_with_values: dict,
    depth: int,
    lt_mode: str,
    verbose: bool,
    outfile: str,
    reverse: bool
):
    pairs = list(entities_with_values.items())
    norm_pairs = tools.minmax_scale_pairs(pairs)

    root = windower(norm_pairs, depth)

    if verbose:
        print("Raw entity values:", pairs)
        print("Normalized values:", [(e, round(v, 3)) for e, v in norm_pairs])
        print("\nWindows:")
        print_windows(root)

    build_kg(
        root=root,
        norm_pairs=norm_pairs,
        outfile=outfile,
        lt_mode=lt_mode,
        reverse=reverse
    )


# ================ MAIN SCRIPT ================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run windower pipeline on entities.")
    parser.add_argument("--n_entities", type=int, default=10)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--D", type=str, default="uniform",
                        choices=["uniform", "normal"])
    parser.add_argument("--low", type=float, default=0.0)
    parser.add_argument("--high", type=float, default=10.0)
    parser.add_argument("--verbose", type=lambda x: str(x).lower() == "true",
                        default=False, choices=[True, False])
    parser.add_argument("--precision", type=int, default=None)
    parser.add_argument("--lt_mode", type=str, required=True,
                        choices=["pairwise", "sequential", "rand5", "pairwise5", "pairwise10"])
    parser.add_argument("--outfile", type=str, required=False)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--mean_e3", action="store_true")

    args = parser.parse_args()

    # ===== Generate entities =====
    entity_array = tools.gen_entity_array(args.n_entities)
    print(f"[INFO] Entities: {entity_array[:10]} ...")

    entities_with_rand_nums = {}

    if args.mean_e3:
        i = 0
        while i < args.n_entities:
            v1 = tools.random_from_distribution(args.D, args.low, args.high, args.precision)
            v2 = tools.random_from_distribution(args.D, args.low, args.high, args.precision)
            v3 = (v1 + v2) / 2

            entities_with_rand_nums[entity_array[i]] = v1
            if i+1 < args.n_entities:
                entities_with_rand_nums[entity_array[i+1]] = v2
            if i+2 < args.n_entities:
                entities_with_rand_nums[entity_array[i+2]] = v3

            i += 3

    else:
        for entity in entity_array:
            entities_with_rand_nums[entity] = tools.random_from_distribution(
                args.D, args.low, args.high, args.precision
            )

    # Generate default outfile name
    if not args.outfile:
        e3 = "e3" if args.mean_e3 else ""
        outfile = f"E-{args.n_entities}_{args.D}_d-{args.depth}_p-{args.precision}_{args.lt_mode}_{e3}.tsv"
    else:
        outfile = args.outfile

    # ===== Run pipeline =====
    run_windower_pipeline(
        entities_with_rand_nums,
        depth=args.depth,
        lt_mode=args.lt_mode,
        verbose=args.verbose,
        outfile=outfile,
        reverse=args.reverse
    )

    print(f"[DONE] KG written to {KG_DIR}")
