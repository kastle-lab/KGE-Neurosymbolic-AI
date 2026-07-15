from __future__ import annotations

import csv
import math
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


Triple = tuple[str, str, str]


def read_kg(tsv_path: str | Path) -> list[Triple]:
    """Read and validate a three-column TSV knowledge graph."""
    path = Path(tsv_path)

    if not path.exists():
        raise FileNotFoundError(f"KG file not found: {path}")

    triples: list[Triple] = []

    with path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            stripped = line.rstrip("\r\n")

            if not stripped:
                continue

            parts = stripped.split("\t")

            if len(parts) != 3:
                raise ValueError(
                    f"Malformed triple in {path} at line {line_number}: "
                    f"expected 3 tab-separated values, found {len(parts)}."
                )

            triples.append((parts[0], parts[1], parts[2]))

    if not triples:
        raise ValueError(f"KG contains no triples: {path}")

    return triples


def write_kg(triples: Iterable[Triple], output_path: str | Path) -> None:
    """Write triples to a UTF-8 TSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as outfile:
        for head, relation, tail in triples:
            outfile.write(f"{head}\t{relation}\t{tail}\n")


def _matches_window_prefix(entity: str, prefixes: Sequence[str]) -> bool:
    lowered = entity.lower()
    return any(lowered.startswith(prefix.lower()) for prefix in prefixes if prefix)


def discover_window_entities(
    triples: Sequence[Triple],
    window_relation: str = "inWindow",
    window_prefixes: Sequence[str] = ("Window_", "window_", "window"),
) -> set[str]:
    """
    Discover window entities without incorrectly treating ordinary entities that
    merely point into a window as windows themselves.

    Objects of `window_relation` are treated as window nodes. Prefix matching is
    also used as a fallback for window nodes that participate in other relations.
    """
    window_entities = {
        entity
        for head, _, tail in triples
        for entity in (head, tail)
        if _matches_window_prefix(entity, window_prefixes)
    }

    for _, relation, tail in triples:
        if relation == window_relation:
            window_entities.add(tail)

    return window_entities


def remove_windows(
    input_path: str | Path,
    output_path: str | Path,
    window_relation: str = "inWindow",
    window_prefixes: Sequence[str] = ("Window_", "window_", "window"),
) -> dict[str, int]:
    """
    Create a windowless copy of a KG.

    The function removes:
      * every triple whose relation is `window_relation`; and
      * every triple whose head or tail is a discovered window entity.

    It returns counts that the pipeline can use for validation and logging.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    triples = read_kg(input_path)
    window_entities = discover_window_entities(
        triples,
        window_relation=window_relation,
        window_prefixes=window_prefixes,
    )

    retained: list[Triple] = []
    removed = 0

    for triple in triples:
        head, relation, tail = triple

        if (
            relation == window_relation
            or head in window_entities
            or tail in window_entities
        ):
            removed += 1
            continue

        retained.append(triple)

    if removed == 0:
        raise ValueError(
            "No window triples were removed. Check --window-relation and "
            "--window-prefix values against the KG encoding."
        )

    write_kg(retained, output_path)

    result = {
        "original_triples": len(triples),
        "window_entities": len(window_entities),
        "window_triples_removed": removed,
        "triples_retained": len(retained),
    }

    print(f"Windowed source KG:       {input_path}")
    print(f"Windowless output KG:     {output_path}")
    print(f"Window entities found:    {result['window_entities']}")
    print(f"Window triples removed:   {result['window_triples_removed']}")
    print(f"Triples retained:         {result['triples_retained']}")

    return result


def _target_relation_triples(
    triples: Sequence[Triple],
    relation_string: str,
) -> list[Triple]:
    return [triple for triple in triples if triple[1] == relation_string]


def _validate_shared_target_relation(
    with_windows_triples: Sequence[Triple],
    without_windows_triples: Sequence[Triple],
    relation_string: str,
) -> list[Triple]:
    with_targets = _target_relation_triples(
        with_windows_triples,
        relation_string,
    )
    without_targets = _target_relation_triples(
        without_windows_triples,
        relation_string,
    )

    if not with_targets:
        raise ValueError(
            f"No triples with relation {relation_string!r} were found."
        )

    if Counter(with_targets) != Counter(without_targets):
        raise ValueError(
            f"The with-windows and without-windows baseline KGs do not contain "
            f"the same {relation_string!r} triples. The paired experiment would "
            "not be controlled."
        )

    duplicates = [
        triple
        for triple, count in Counter(with_targets).items()
        if count > 1
    ]

    if duplicates:
        raise ValueError(
            f"Duplicate {relation_string!r} triples were found. Exact paired "
            "removal is ambiguous until duplicates are removed."
        )

    return with_targets


def _validate_percentages(percentages: Sequence[float]) -> list[float]:
    if not percentages:
        raise ValueError("At least one removal percentage is required.")

    normalized = [float(value) for value in percentages]

    if len(set(normalized)) != len(normalized):
        raise ValueError("Removal percentages must be unique.")

    for percentage in normalized:
        if percentage <= 0 or percentage > 100:
            raise ValueError(
                f"Removal percentages must be greater than 0 and at most 100. "
                f"Received {percentage}."
            )

    return normalized


def _half_up_count(total: int, percentage: float) -> int:
    return int(math.floor((total * percentage / 100.0) + 0.5))


def _write_removal_plan(
    plan_path: Path,
    relation_string: str,
    ordered_targets: Sequence[Triple],
    percentages: Sequence[float],
    removal_sets: dict[float, set[Triple]],
    seed: int,
    nested: bool,
) -> None:
    plan_path.parent.mkdir(parents=True, exist_ok=True)

    with plan_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=[
                "removal_rank",
                "head",
                "relation",
                "tail",
                "first_removed_percent",
                "seed",
                "nested_removals",
            ],
        )
        writer.writeheader()

        for rank, triple in enumerate(ordered_targets, start=1):
            first_removed_percent = None

            for percentage in sorted(percentages):
                if triple in removal_sets[percentage]:
                    first_removed_percent = percentage
                    break

            writer.writerow(
                {
                    "removal_rank": rank,
                    "head": triple[0],
                    "relation": relation_string,
                    "tail": triple[2],
                    "first_removed_percent": first_removed_percent,
                    "seed": seed,
                    "nested_removals": nested,
                }
            )


def create_paired_percentage_variants(
    with_windows_path: str | Path,
    without_windows_path: str | Path,
    percentages: Sequence[float],
    with_windows_output_paths: Sequence[str | Path],
    without_windows_output_paths: Sequence[str | Path],
    relation_string: str = "hasAge",
    seed: int = 42,
    nested: bool = True,
    removal_plan_path: str | Path | None = None,
) -> dict[float, set[Triple]]:
    """
    Create paired percentage-removal variants for both window conditions.

    At each percentage, the exact same target-relation triples are removed from
    the with-windows and without-windows KGs. With nested=True, every removal at
    a lower percentage remains removed at all higher percentages.
    """
    percentages = _validate_percentages(percentages)

    if len(percentages) != len(with_windows_output_paths):
        raise ValueError(
            "percentages and with_windows_output_paths must have equal lengths."
        )

    if len(percentages) != len(without_windows_output_paths):
        raise ValueError(
            "percentages and without_windows_output_paths must have equal lengths."
        )

    all_output_paths = [
        Path(path)
        for path in [
            *with_windows_output_paths,
            *without_windows_output_paths,
        ]
    ]

    if len(set(all_output_paths)) != len(all_output_paths):
        raise ValueError("All paired variant output paths must be unique.")

    with_windows_triples = read_kg(with_windows_path)
    without_windows_triples = read_kg(without_windows_path)

    target_triples = _validate_shared_target_relation(
        with_windows_triples,
        without_windows_triples,
        relation_string,
    )

    total_targets = len(target_triples)
    removal_sets: dict[float, set[Triple]] = {}

    if nested:
        rng = random.Random(seed)
        ordered_targets = target_triples.copy()
        rng.shuffle(ordered_targets)

        for percentage in percentages:
            count = _half_up_count(total_targets, percentage)
            removal_sets[percentage] = set(ordered_targets[:count])
    else:
        ordered_targets = target_triples.copy()

        for index, percentage in enumerate(percentages, start=1):
            count = _half_up_count(total_targets, percentage)
            rng = random.Random(seed + index)
            removal_sets[percentage] = set(
                rng.sample(target_triples, count)
            )

    for percentage, with_output, without_output in zip(
        percentages,
        with_windows_output_paths,
        without_windows_output_paths,
    ):
        removed = removal_sets[percentage]

        with_retained = [
            triple
            for triple in with_windows_triples
            if triple not in removed
        ]
        without_retained = [
            triple
            for triple in without_windows_triples
            if triple not in removed
        ]

        write_kg(with_retained, with_output)
        write_kg(without_retained, without_output)

        actual_removed = len(removed)
        actual_percent = actual_removed / total_targets * 100.0

        print()
        print(f"Created paired {percentage:g}% variants")
        print(f"  With windows:           {Path(with_output)}")
        print(f"  Without windows:        {Path(without_output)}")
        print(f"  {relation_string} removed: {actual_removed}/{total_targets}")
        print(f"  Actual percentage:      {actual_percent:.2f}%")

    if removal_plan_path is not None:
        _write_removal_plan(
            plan_path=Path(removal_plan_path),
            relation_string=relation_string,
            ordered_targets=ordered_targets,
            percentages=percentages,
            removal_sets=removal_sets,
            seed=seed,
            nested=nested,
        )
        print(f"Saved shared removal plan: {Path(removal_plan_path)}")

    return removal_sets


def modify_kg(path: str, n: int, relation_string: str) -> None:
    """Legacy helper that removes every nth occurrence of a relation."""
    if n < 1:
        raise ValueError("n must be at least 1.")

    directory = Path(path)
    source = directory / "kg.tsv"
    output = directory / f"every_{n}_removed_kg.tsv"

    triples = read_kg(source)
    retained: list[Triple] = []
    counter = 0
    total_removed = 0

    for triple in triples:
        if triple[1] == relation_string:
            counter += 1

            if counter == n:
                counter = 0
                total_removed += 1
                continue

        retained.append(triple)

    write_kg(retained, output)
    print(f"Created legacy modified KG: {output}")
    print(f"Removed {relation_string} total: {total_removed}")
