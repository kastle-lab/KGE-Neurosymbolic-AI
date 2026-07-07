from typing import Sequence, Optional, Tuple, List
import os
import sys
import re

import numpy as np


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.environ.get(
    "SLURM_SUBMIT_DIR",
    os.path.dirname(os.path.abspath(__file__)),
)

KG_DIR = os.path.join(BASE_DIR)
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


def create_kg_r(
    n_vertices,
    window_depth,
    experiment_folder,
    decimal_precision,
    high,
    low,
    n_people=0,
    include_people=True,
    include_windowing=True,
):
    """
    Creates a KG where vertex labels are arbitrary, but their ordering
    is determined by random assigned values in [0, 1].

    The kg.tsv.val file is the ground-truth ordering source.

    If include_windowing=False, no inWindow triples are written.
    """

    vertices = [f"v{i}" for i in range(n_vertices)]

    assigned_vertices = {}

    for vertex in vertices:
        assigned_vertices[vertex] = round(
            np.random.uniform(0.0, 1.0),
            decimal_precision,
        )

    names_with_assigned_ages = {}

    if include_people:
        names = [f"person{i}" for i in range(n_people)]

        for person in names:
            names_with_assigned_ages[person] = round(
                np.random.uniform(0.0, 1.0),
                decimal_precision,
            )

    pairs = list(assigned_vertices.items())

    # Sort before windowing so windows reflect value ordering.
    norm_pairs = sorted(pairs, key=lambda x: x[1])

    root = None
    if include_windowing:
        root = windower(norm_pairs, window_depth)

    build_kg(
        root=root,
        norm_pairs=norm_pairs,
        experiment_folder=experiment_folder,
        names_with_assigned_ages=names_with_assigned_ages,
        include_people=include_people,
        include_windowing=include_windowing,
    )


def create_kg(
    n_vertices,
    window_depth,
    experiment_folder,
    decimal_precision,
    high,
    low,
    n_people=0,
    include_people=True,
    include_windowing=True,
):
    """
    Creates a KG for the experiments.

    Writes:
        - lessThan relationships among property/value nodes
        - hasAge relationships from people to value nodes, if include_people=True
        - hasAgeLessThan relationships among people, if include_people=True
        - optionally inWindow relationships, if include_windowing=True
    """

    # Generate vertices named v<low> ... v<high>.
    vertices = [f"v{i}" for i in range(low, high + 1)]

    assigned_vertices = {}

    for vertex in vertices:
        assigned_vertices[vertex] = int(vertex[1:])

    names_with_assigned_ages = {}

    if include_people:
        names = [f"person{i}" for i in range(n_people)]

        for person in names:
            age = np.random.uniform(low, high)
            names_with_assigned_ages[person] = round(age, decimal_precision)

    pairs = list(assigned_vertices.items())
    norm_pairs = pairs

    root = None
    if include_windowing:
        root = windower(norm_pairs, window_depth)

    build_kg(
        root=root,
        norm_pairs=norm_pairs,
        experiment_folder=experiment_folder,
        names_with_assigned_ages=names_with_assigned_ages,
        include_people=include_people,
        include_windowing=include_windowing,
    )


def build_kg(
    root: Optional[Window],
    norm_pairs: List[Tuple[str, float]],
    experiment_folder: str,
    names_with_assigned_ages=None,
    include_people=True,
    include_windowing=True,
):
    """
    Writes the KG.

    If include_windowing=False, skips all inWindow triples.
    """

    if names_with_assigned_ages is None:
        names_with_assigned_ages = {}

    exp_dir = os.path.join(KG_DIR, experiment_folder)
    os.makedirs(exp_dir, exist_ok=True)

    out_path = os.path.join(exp_dir, "kg.tsv")
    vals_path = f"{out_path}.val"

    print(f"[INFO] Writing KG to: {out_path}")
    print(f"[INFO] Writing values to: {vals_path}")

    if include_windowing:
        print("[INFO] Including windowing triples.")
    else:
        print("[INFO] Skipping windowing triples.")

    if include_people:
        print("[INFO] Including person nodes, hasAge, and hasAgeLessThan relations.")
    else:
        print("[INFO] Property-only mode: no person nodes, hasAge, or hasAgeLessThan relations.")

    with open(out_path, "w") as f, open(vals_path, "w") as vf:

        # --------- Write window triples ---------
        if include_windowing:
            if root is None:
                raise ValueError("include_windowing=True but root is None.")

            def add_window_triples(node: Window):
                window_label = f"Window_{node.path.replace('->', '_')}"

                for e, _ in node.elements:
                    f.write(f"{e}\tinWindow\t{window_label}\n")

                if node.left:
                    add_window_triples(node.left)
                if node.right:
                    add_window_triples(node.right)

            add_window_triples(root)

        # --------- Sort value/property nodes ---------
        sorted_pairs = sorted(norm_pairs, key=lambda x: x[1])
        print(f"[DEBUG] First 10 sorted pairs: {sorted_pairs[:10]}")

        # --------- Write hasAge and lessThan edges ---------
        for i, (e1, value) in enumerate(sorted_pairs):

            if include_people:
                for person, age in names_with_assigned_ages.items():
                    if age == value:
                        f.write(f"{person}\thasAge\t{e1}\n")

            for e2, value2 in sorted_pairs[i + 1:]:
                f.write(f"{e1}\tlessThan\t{e2}\n")

        # --------- Write values ---------
        for e, v in sorted_pairs:
            vf.write(f"{e}\t{v}\n")

        if include_people:
            # --------- Write less-than relationships among people ---------
            write_person_lt_rels = True

            if write_person_lt_rels:
                sorted_people = sorted(
                    names_with_assigned_ages.items(),
                    key=lambda x: x[1],
                )

                for i, (person1, age1) in enumerate(sorted_people):
                    for person2, age2 in sorted_people[i + 1:]:
                        if age1 < age2:
                            f.write(f"{person1}\thasAgeLessThan\t{person2}\n")

            # --------- Optional same-age relationships among people ---------
            write_person_same_age_rels = False

            if write_person_same_age_rels:
                people_list = list(names_with_assigned_ages.items())

                for i, (person1, age1) in enumerate(people_list):
                    for person2, age2 in people_list[i + 1:]:
                        if age1 == age2:
                            f.write(f"{person1}\thasSameAge\t{person2}\n")
                            f.write(f"{person2}\thasSameAge\t{person1}\n")


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

    values = np.array([v for _, v in pairs], dtype=float)
    mn, mx = values.min(), values.max()

    precisions = [infer_precision(v) for _, v in pairs]
    avg_precision = int(round(np.mean(precisions))) if precisions else 0

    if mn == mx:
        normed = np.zeros_like(values)
    else:
        normed = (values - mn) / (mx - mn)

    return [
        (e, round(float(v), avg_precision))
        for (e, _), v in zip(pairs, normed)
    ]


def infer_precision(val: float) -> int:
    """
    Infer number of decimal places from a float as written.
    """

    s = f"{val}"

    if "e" in s or "E" in s:
        s = f"{val:.16f}".rstrip("0")

    match = re.search(r"\.(\d+)", s)

    return len(match.group(1)) if match else 0


def split_pairs(pairs: Sequence[Tuple[str, float]]):
    mid = len(pairs) // 2
    return pairs[:mid], pairs[mid:]


if __name__ == "__main__":
    # Example with windowing.
    # create_kg(
    #     n_vertices=100,
    #     window_depth=4,
    #     experiment_folder="100people_with_windows",
    #     decimal_precision=0,
    #     high=100,
    #     low=1,
    #     n_people=100,
    #     include_people=True,
    #     include_windowing=True,
    # )

    # Example without windowing.
    create_kg(
        n_vertices=100,
        window_depth=4,
        experiment_folder="100people_no_windows",
        decimal_precision=0,
        high=100,
        low=1,
        n_people=100,
        include_people=True,
        include_windowing=False,
    )