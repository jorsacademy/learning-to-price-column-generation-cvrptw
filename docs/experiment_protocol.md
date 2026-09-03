# Experiment Protocol

## Data separation

Training, validation, and test seed ranges are disjoint. Validation splitting is performed by complete CVRPTW instance. Arc rows and pricing iterations from one instance therefore remain in the same split.

## Training labels

At each exact column-generation iteration, the top-k routes returned by full pricing define positive arcs. Labels are the union of route arcs, including depot departure and return arcs. Depot arcs are retained at inference regardless of score; the model controls customer-to-customer transitions.

## Baselines

Every benchmark should include:

- full exact pricing;
- handcrafted dual-aware arc pruning;
- matched-density random pruning;
- learned pruning;
- full route enumeration where tractable.

Random pruning is repeated through deterministic per-iteration seeds. It controls for the computational effect of graph sparsification itself.

## Metrics

Report classification and optimization metrics separately.

Classification:

- precision;
- recall;
- F1;
- positive rate.

Pricing and column generation:

- arc retention;
- labels created and dominated;
- restricted-pricing success rate;
- exact fallback count;
- exact pricing time;
- restricted pricing time;
- RMP iterations;
- final LP objective;
- global certificate status.

Integer recovery:

- restricted integer-master objective;
- full enumerated integer objective on tiny cases;
- integer gap where the full route set is available.

## Distribution shifts

The frozen protocol separates:

- same-size uniform/wide-window testing;
- larger-customer-count testing;
- clustered-coordinate testing;
- tighter-time-window testing.

Do not pool these into one number without retaining scenario-level results.

## Claims

A learned selector should not be called faster because it has high F1. Runtime, label count, fallback rate, and total column-generation work must be measured directly. Report overhead and unfavorable seeds.
