from typing import Sequence, Optional, Tuple, List, Literal
import random
import os, sys, re
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.environ.get("SLURM_SUBMIT_DIR", os.path.dirname(os.path.abspath(__file__)))

#SCRATCH = os.environ.get("SCRATCH")
KG_DIR = os.path.join(BASE_DIR, "version2")
os.makedirs(KG_DIR, exist_ok=True)
    
class Window:
    def __init__(self, elements: Sequence[Tuple[str, float]], n: int, path="root"):
        self.elements: List[Tuple[str, float]] = list(elements)
        self.n = n
        self.path = path
        self.left: Optional["Window"] = None
        self.right: Optional["Window"] = None

    def __repr__(self):
        return f"Window(path={self.path}, n={self.n}, size={len(self.elements)})"

def create_kg(n_vertices, window_depth, experiment_folder, decimal_precision, high, low, n_people):
    """
    Creates a KG for our experiments.

    Args:
        assigned_vertices: labeled vertices and their assigned values
        window_depth: (int) the depth of binary windowing
    
    Outputs a tsv file
    """
    
    # generate an array of n vertices named V0, V1, ... Vn-1.
    vertices = [f"v{i}" for i in range(n_vertices)]
    
    assigned_vertices = {}
    
    # assign random numbers to each vertex
    for i, vertex in enumerate(vertices):
        assigned_vertices[vertex] = i
        #assigned_vertices[vertex] = round(assigned_vertices[vertex], decimal_precision)
        

    # create 5000 people
    names = [f"person{i}" for i in (range(n_people))]
    names_with_assigned_ages = {}
    
    # assign random ages to each name
    for person in names:
        names_with_assigned_ages[person] = np.random.uniform(low, high)
        names_with_assigned_ages[person] = round(names_with_assigned_ages[person], decimal_precision)

    # sort nodes to build windowing relationships
    pairs = list(assigned_vertices.items())
    norm_pairs = pairs # minmax_scale_pairs(pairs)

    root = windower(norm_pairs, window_depth)
        
    build_kg(
        root=root,
        norm_pairs=norm_pairs,
        experiment_folder=experiment_folder,
        names_with_assigned_ages=names_with_assigned_ages
    )
    
def build_kg(root: Window, norm_pairs: List[Tuple[str, float]], experiment_folder: str, names_with_assigned_ages):

    exp_dir = os.path.join(KG_DIR, experiment_folder)
    os.makedirs(exp_dir, exist_ok=True)

    out_path = os.path.join(exp_dir, "kg.tsv")
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
        sorted_pairs = sorted(norm_pairs, key=lambda x: x[1])
        print(f"[DEBUG] First 10 sorted pairs: {sorted_pairs[:10]}")

        # --------- Write hasAge edges ---------
        for i, (e1, value) in enumerate(sorted_pairs):
            
            for person, age in names_with_assigned_ages.items():
                if age == value:
                    f.write(f"{person}\thasAge\t{e1}\n")

            
            for e2, value in sorted_pairs[i+1:]:
                f.write(f"{e1}\tlessThan\t{e2}\n")

        # --------- Write values ---------
        for e, v in sorted_pairs:
            vf.write(f"{e}\t{v}\n")
            
            
def windower(pairs: Sequence[Tuple[str, float]], n: int = 1, path="root") -> Window:
    node = Window(pairs, n, path=path)
    if n <= 1 or len(pairs) <= 1:
        return node
    left, right = split_pairs(pairs)
    node.left = windower(left, n - 1, path + "->L")
    node.right = windower(right, n - 1, path + "->R")
    return node

def minmax_scale_pairs(pairs: Sequence[Tuple[str, float]]):
    if not pairs:
        return []

    # extract numeric values
    values = np.array([v for _, v in pairs], dtype=float)
    mn, mx = values.min(), values.max()

    # infer average decimal precision across inputs
    precisions = [infer_precision(v) for _, v in pairs]
    avg_precision = int(round(np.mean(precisions))) if precisions else 0

    # normalize
    if mn == mx:
        normed = np.zeros_like(values)
    else:
        normed = (values - mn) / (mx - mn)

    # round normalized values to the same precision
    return [(e, round(float(v), avg_precision)) for (e, _), v in zip(pairs, normed)]

def infer_precision(val: float) -> int:
    """Infer number of decimal places from a float as written."""
    s = f"{val}"
    if "e" in s or "E" in s:
        # handle scientific notation by expanding it
        s = f"{val:.16f}".rstrip("0")
    match = re.search(r"\.(\d+)", s)
    return len(match.group(1)) if match else 0

def split_pairs(pairs: Sequence[Tuple[str, float]]):
    mid = len(pairs) // 2
    return pairs[:mid], pairs[mid:]