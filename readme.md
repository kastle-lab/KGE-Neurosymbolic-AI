## Repository Structure

The readme describes the pipeline inside `experiment-final` and its imports. The files inside `old-plots-and-code` though important to motivating the research question through visualizations (PCA) of various graph embeddings, are not directly pertinent to the experiment and its results.

## Experiment Overview

Knowledge graph embeddings learn vector representations from the structure of a graph. Most work on knowledge graph embeddings focuses on predicting missing entities or relationships. This project instead explores a related question:

> How does the topology of a knowledge graph affect the numerical information preserved in its learned embeddings?

The experiments examine whether age information remains recoverable after the graph is structurally modified and increasing numbers of explicit `hasAge` relationships are removed.

Window relationships are the primary structural modification examined in the current study. A graph containing this additional structure is compared with a corresponding graph in which the window-related triples have been removed.

The project is exploratory rather than intended as a broad benchmark. The current experiments use:

- The **MuRE** embedding model
- Synthetic graphs containing **100, 200, and 500 people**
- Numeric age values connected through `hasAge`
- Windowed and windowless graph structures
- `hasAge` removal levels of **0%, 15%, 30%, 45%, 60%, and 75%**
- Query-point and learned-regression recovery methods

The broader goal is to empirically investigate what happens to the information represented in embedding space when a controlled mathematical change is made to the graph structure.

---

## Experimental Design

Each synthetic knowledge graph contains people connected to numeric age values through `hasAge` relationships.

```text
person1  hasAge  age24
person2  hasAge  age37
person3  hasAge  age51