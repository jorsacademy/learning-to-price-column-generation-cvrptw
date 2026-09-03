# Learning to Price Column Generation for CVRPTW

[![CI](https://github.com/jorsacademy/learning-to-price-column-generation-cvrptw/actions/workflows/ci.yml/badge.svg)](https://github.com/jorsacademy/learning-to-price-column-generation-cvrptw/actions/workflows/ci.yml)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-orange)](LICENSE)

A research-oriented implementation of **machine-learning-assisted pricing inside route-based column generation** for the Capacitated Vehicle Routing Problem with Time Windows (CVRPTW).

The central design principle is conservative:

> Learning is allowed to accelerate the search for improving columns, but it is not allowed to certify column-generation convergence.

A dual-aware neural arc scorer prunes the pricing graph. The resulting restricted Elementary Shortest Path Problem with Resource Constraints (ESPPRC) is solved exactly on that graph. When the restricted graph produces no negative-reduced-cost route, the implementation automatically solves the full pricing problem. The LP algorithm stops only after that full-graph solve certifies that no improving route remains.

## Research question

Can a learned arc selector reduce pricing work while preserving the final route-based LP bound through exact fallback?

The repository measures more than classifier accuracy. It reports:

- column-generation iterations;
- generated route-pool size;
- exact fallback frequency;
- retained arc fraction;
- label counts and pricing time;
- final LP objective and certification status;
- restricted integer-master objective;
- exact LP and integer gaps where full route enumeration is tractable;
- generalization under customer-count, spatial-distribution, and time-window shifts.

No speedup claim is hard-coded. The experiment is designed to retain negative results, including cases where model inference or poor pruning increases total work.

## Optimization model

Let `R` denote the set of feasible elementary depot-to-depot routes and `a_ir` indicate whether route `r` serves customer `i`. The route cost `c_r` contains Euclidean travel distance and a fixed vehicle-use cost.

The set-partitioning master LP is

```text
minimize    sum_r c_r x_r
subject to  sum_r a_ir x_r = 1       for every customer i
            x_r >= 0.
```

For customer dual values `pi_i`, route `r` has reduced cost

```text
c_bar_r = c_r - sum_i pi_i a_ir.
```

Pricing seeks a capacity- and time-window-feasible elementary route with minimum reduced cost. A route with `c_bar_r < 0` is added to the restricted master.

The repository assumes an unlimited homogeneous fleet. A fleet-size row, multiple depots, heterogeneous vehicles, pickup-and-delivery constraints, and branch-and-price are outside version 0.1.

## Architecture

```text
Synthetic or loaded CVRPTW instance
                 │
                 ▼
       singleton initial routes
                 │
                 ▼
       restricted master LP
        SciPy linprog / HiGHS
                 │
                 ├── customer duals ──────────────────────────────┐
                 │                                                │
                 ▼                                                ▼
       learned / heuristic                              exact full-graph
       arc selection                                    label-setting ESPPRC
                 │                                                ▲
                 ▼                                                │
 exact pricing on restricted graph                              fallback
                 │                                                │
        negative route found? ── no ──────────────────────────────┘
                 │ yes
                 ▼
            add column
                 │
                 └────────────── repeat
```

## What is learned?

The training pipeline runs exact column generation on disjoint synthetic training instances. At every restricted-master iteration it records:

1. the customer dual vector;
2. one feature vector for each directed depot/customer arc;
3. the union of arcs appearing in the exact top-k priced routes.

The resulting edge-classification dataset contains 17 transparent features, including:

- normalized travel distance;
- origin and target distance to the depot;
- Clarke–Wright-style saving;
- origin and target dual values;
- target-dual-minus-distance attractiveness;
- capacity fractions;
- time-window positions and widths;
- service-time fraction;
- depot indicators;
- a coarse static time-feasibility flag.

A compact two-layer ReLU MLP is implemented directly in NumPy and trained with weighted binary cross entropy and Adam. This keeps the baseline auditable and avoids a heavyweight runtime dependency. It is not presented as a reproduction of an attention/RL pricing paper.

Training/validation splitting is performed by **whole problem instance**, not by arc row. Therefore pricing iterations from the same instance cannot leak across the split.

## Pricing methods

| Mode | First pricing attempt | Exact fallback | Final LP certificate |
| --- | --- | --- | --- |
| `exact` | Full label-setting ESPPRC | Not applicable | Yes |
| `heuristic` | Handcrafted dual-aware arc pruning | Required on failure | Yes |
| `random` | Matched-density random arc pruning | Required on failure | Yes |
| `learned` | MLP-scored arc pruning | Required on failure | Yes |

The random policy is an important control. It separates gains caused by learned structure from gains caused merely by reducing graph density.

## Exactness contract

For the declared LP relaxation:

1. every route inserted into the master is independently checked for capacity and time-window feasibility;
2. the full pricing implementation is validated against complete feasible-route enumeration on tiny cases;
3. the exact column-generation LP objective is tested against the full enumerated route master;
4. heuristic and learned modes may add any negative-reduced-cost column, not necessarily the most negative one;
5. failure of restricted pricing always triggers full exact pricing;
6. convergence is declared only when full exact pricing finds no negative-reduced-cost route.

Accordingly, learned pruning can affect iteration count and runtime, but it cannot by itself produce a false LP convergence certificate.

The binary master solved over the generated route pool is reported as a **restricted integer master**. It is not a global CVRPTW optimality proof unless every feasible route has been enumerated. See [`docs/exactness.md`](docs/exactness.md).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Core runtime dependencies are NumPy and SciPy.

## Quick start

Run exact column generation on a generated instance:

```bash
learning-to-price demo --customers 8 --seed 42 --mode exact
```

Compare a handcrafted restricted-pricing heuristic with exact fallback:

```bash
learning-to-price demo \
  --customers 10 \
  --seed 42 \
  --mode heuristic \
  --keep-ratio 0.35
```

### 1. Collect exact pricing labels

```bash
learning-to-price collect \
  --instances 24 \
  --customers 8 \
  --seed 1000 \
  --top-k-routes 3 \
  --output artifacts/pricing-dataset.npz
```

### 2. Train the arc scorer

```bash
learning-to-price train \
  artifacts/pricing-dataset.npz \
  --epochs 80 \
  --hidden-dim 32 \
  --checkpoint artifacts/arc-scorer.npz
```

### 3. Benchmark learned pricing

```bash
learning-to-price benchmark \
  --checkpoint artifacts/arc-scorer.npz \
  --instances 12 \
  --customers 9 \
  --seed 20000 \
  --output-json artifacts/benchmark.json \
  --output-csv artifacts/benchmark.csv
```

The benchmark compares exact, handcrafted, matched-density random, and learned pricing on identical instances.

## Frozen research protocol

The end-to-end research script trains on uniform, wide-window instances and evaluates four disjoint scenarios:

1. in-distribution;
2. customer-count shift;
3. clustered-coordinate shift;
4. tighter-time-window shift.

```bash
python scripts/research_experiment.py \
  --train-instances 24 \
  --train-customers 8 \
  --epochs 80 \
  --benchmark-instances 8 \
  --keep-ratio 0.35
```

Defaults are documented in [`configs/research_v1.json`](configs/research_v1.json) and the methodological rules are described in [`docs/experiment_protocol.md`](docs/experiment_protocol.md).

## Repository structure

```text
src/learning_to_price/
├── domain.py             typed CVRPTW data model and JSON I/O
├── generator.py          reproducible Solomon-like synthetic instances
├── geometry.py           Euclidean travel matrix
├── routes.py             route feasibility and finite route enumeration
├── master.py             restricted LP and fixed-pool integer masters
├── pricing.py            exact label-setting ESPPRC and brute-force oracle
├── features.py           dual-aware arc features and pruning policies
├── learning.py           NumPy MLP, Adam, metrics, checkpointing
├── dataset.py            exact-CG trajectory collection and grouped split
├── column_generation.py  exact and safe hybrid CG loops
├── benchmark.py          repeated-instance evaluation and reports
├── experiment.py         in-distribution and shift protocol
└── cli.py                command-line interface
```

## Tests and CI

```bash
ruff check .
ruff format --check .
mypy src
pytest --cov=learning_to_price --cov-report=term-missing
```

The regression suite checks:

- deterministic feasible-instance generation;
- capacity and asymmetric time-window feasibility;
- exact label setting against full route enumeration;
- exact column-generation LP against the full route master;
- mandatory fallback and global certification in hybrid modes;
- restricted integer-master correctness on a hand-checkable case;
- grouped train/validation splitting;
- MLP loss reduction and checkpoint round trip;
- learned-hybrid equality with the exact LP bound;
- benchmark and CLI smoke paths.

GitHub Actions runs Python 3.11 and 3.12. It installs the package, checks formatting and linting, performs strict type checking, runs branch-aware coverage, and executes a small end-to-end collect/train/learned-pricing smoke experiment without external services.

## Methodological boundaries

This repository does **not** claim:

- industrial CVRPTW scale;
- branch-and-price or integer optimality for the general problem;
- paper-level parity with learned pricing methods;
- that edge-classification accuracy implies pricing speedup;
- that synthetic training generalizes to Solomon or industrial datasets;
- that the compact NumPy MLP is a state-of-the-art architecture;
- that every restricted pricing success is the globally most negative route.

The exact label-setting state includes the visited-customer mask and is intended for small research instances. Scaling to large benchmarks requires relaxations and engineering such as ng-routes, decremental state-space relaxation, bidirectional labeling, dominance strengthening, stabilization, and branch-price-and-cut integration.

## Research context

This implementation is positioned between two lines of work:

- ML-assisted graph reduction for constrained-shortest-path pricing, where learned arc relevance reduces the pricing network;
- end-to-end learned pricing policies, including recent attention/RL approaches.

The repository follows the first line and adds a strict exact-fallback contract. It uses the second line as a future research direction rather than claiming an implementation of it. See [`docs/research_context.md`](docs/research_context.md).

## References

1. Morabit, M., Desaulniers, G., & Lodi, A. (2023). Machine-Learning–Based Arc Selection for Constrained Shortest Path Problems in Column Generation. *INFORMS Journal on Optimization*, 5(2), 191–210. https://doi.org/10.1287/ijoo.2022.0082
2. Abouelrous, A., Bliek, L., Gabor, A. F., Wu, Y., & Zhang, Y. (2025). Reinforcement Learning for Solving the Pricing Problem in Column Generation: Applications to Vehicle Routing. arXiv:2504.02383. https://arxiv.org/abs/2504.02383
3. Mandal, U., Regan, A., Rousseau, L.-M., & Yarkony, J. (2023). Graph Master and Local Area Routes for Efficient Column Generation for the Capacitated Vehicle Routing Problem with Time Windows. arXiv:2304.11723. https://arxiv.org/abs/2304.11723
4. Cordeau, J.-F., Desaulniers, G., Desrosiers, J., Solomon, M. M., & Soumis, F. (2007). VRP with Time Windows. In *The Vehicle Routing Problem*. SIAM.

## License

Licensed under **PolyForm Noncommercial 1.0.0**. Commercial use is not granted. The project is source-available, not OSI Open Source. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
