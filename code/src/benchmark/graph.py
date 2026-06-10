"""The PGxLOD graph, read from the CSV of the extraction.

Files read in ``csv_dir``:

    entities_{drug,genetic,phenotype}.csv   uri, name, type
    ucpgx_components.csv                    ucpgx, predicate, object
    ucpgx_links_unique.csv                  a, relation, b, oriente

The gold is read from ``ucpgx_links_unique.csv``, where the duplicates are already removed
and ``narrowMatch`` is folded into an oriented ``broadMatch``. A UCPGx has no name of its
own: its content is its components and the predicates that link it to them.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# pgxo class of a component to its coarse role
ROLE = {
    "Drug": "drug",
    "Gene": "genetic", "Variant": "genetic", "GenomicVariation": "genetic",
    "Haplotype": "genetic", "GeneticFactor": "genetic",
    "Phenotype": "phenotype", "Disease": "phenotype",
    "PharmacodynamicPhenotype": "phenotype", "PharmacokineticPhenotype": "phenotype",
}


@dataclass
class Graph:
    """The graph in memory.

    ``name``      uri -> label (components only)
    ``ntype``     uri -> pgxo class (Drug, Variant, Phenotype, ...)
    ``comp``      ucpgx -> list of (predicate, object)
    ``links``     relation -> list of (a, b), the gold
    ``ucpgx``     sorted URIs of the UCPGx with at least one component
    ``oriented``  relation -> bool
    """

    name: dict = field(default_factory=dict)
    ntype: dict = field(default_factory=dict)
    comp: dict = field(default_factory=dict)
    links: dict = field(default_factory=dict)
    ucpgx: list = field(default_factory=list)
    oriented: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"ucpgx": len(self.ucpgx), "named_entities": len(self.name),
                "relations": {r: len(v) for r, v in self.links.items()},
                "component_edges": sum(len(v) for v in self.comp.values())}


def load_graph(csv_dir: str | Path, logger=None) -> Graph:
    """Build a :class:`Graph` from ``csv_dir``."""
    csv_dir = Path(csv_dir)

    ent = pd.concat(
        [pd.read_csv(csv_dir / f"entities_{g}.csv", low_memory=False)
         for g in ("drug", "genetic", "phenotype")],
        ignore_index=True,
    ).drop_duplicates("uri")
    name = dict(zip(ent["uri"], ent["name"]))
    ntype = dict(zip(ent["uri"], ent["type"]))

    comp_df = pd.read_csv(csv_dir / "ucpgx_components.csv")
    comp: dict = defaultdict(list)
    for u, p, o in zip(comp_df["ucpgx"], comp_df["predicate"], comp_df["object"]):
        comp[u].append((p, o))
    comp = dict(comp)

    links_df = pd.read_csv(csv_dir / "ucpgx_links_unique.csv")
    links, oriented = {}, {}
    for r, sub in links_df.groupby("relation"):
        links[r] = list(zip(sub["a"], sub["b"]))
        oriented[r] = bool(sub["oriente"].iloc[0]) if "oriente" in sub.columns else False

    g = Graph(name=name, ntype=ntype, comp=comp, links=links,
              ucpgx=sorted(comp.keys()), oriented=oriented)
    if logger:
        logger.info(f"graph loaded: {g.summary()}")
    return g


def verbalize(g: Graph, uri: str) -> str:
    """Flat verbalisation of a UCPGx: its named components by role, then its predicates.

    Values are sorted and deduplicated, so two UCPGx linking the same entities give the
    same string. That string is the grouping key of the anti-leak split.
    """
    slots = {"drug": [], "genetic": [], "phenotype": []}
    effect = []
    for pred, obj in g.comp.get(uri, ()):
        role = ROLE.get(g.ntype.get(obj))
        text = g.name.get(obj)
        if role in slots and text:
            slots[role].append(text)
        effect.append(pred)
    parts = []
    for role, label in (("drug", "Drug"), ("genetic", "Genetic factor"),
                        ("phenotype", "Phenotype")):
        vals = sorted(set(slots[role]))
        parts.append(f"{label}: {', '.join(vals) if vals else 'not specified'}")
    parts.append(f"Effect: {', '.join(sorted(set(effect)))}")
    return ". ".join(parts) + "."


def has_genetic(g: Graph, uri: str) -> bool:
    return any(ROLE.get(g.ntype.get(o)) == "genetic" for _, o in g.comp.get(uri, ()))


def pair_family(g: Graph, a: str, b: str) -> str:
    """``pgx`` when both ends carry a genetic component, ``onco`` when neither does,
    ``mixed`` when only one does (96 broadMatch pairs, left out)."""
    ha, hb = has_genetic(g, a), has_genetic(g, b)
    if ha and hb:
        return "pgx"
    if not ha and not hb:
        return "onco"
    return "mixed"


def encode(g: Graph) -> tuple[int, int, np.ndarray]:
    """``(num_nodes, num_relations, triples)``: the graph the models consume.

    UCPGx hold the indices 0..U-1 in the order of ``g.ucpgx``, components the next ones in
    order of first appearance, so the embedding of a UCPGx is read at the row its index
    gives. Relations are numbered in order of first appearance. A triple is a
    ``(ucpgx, predicate, component)`` edge; no alignment link is a triple.
    """
    ucpgx = list(g.ucpgx)
    seen, components = set(ucpgx), []
    for u in ucpgx:
        for _, o in g.comp.get(u, ()):
            if o not in seen:
                seen.add(o)
                components.append(o)
    id_of = {u: i for i, u in enumerate(ucpgx + components)}
    rel_of: dict = {}
    for u in ucpgx:
        for p, _ in g.comp.get(u, ()):
            rel_of.setdefault(p, len(rel_of))
    triples = [(id_of[u], rel_of[p], id_of[o]) for u in ucpgx for p, o in g.comp.get(u, ())]
    return len(id_of), len(rel_of), np.asarray(triples, dtype=np.int64).reshape(-1, 3)
