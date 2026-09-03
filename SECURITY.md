# Security and Reliability

This repository is a research prototype, not a production routing service.

## Resource exhaustion

Elementary pricing and full route enumeration are exponential in the worst case. Do not run untrusted, arbitrarily large instances without external process, memory, and time limits. The exact pricer includes a label cap; full enumeration includes customer and route caps.

## Input validation

Loaded instance data are checked for finite values, ordered time windows, contiguous customer IDs, positive demands, and valid capacity/horizon values. This does not establish that external business data are correct.

## Learned guidance

The MLP produces arc scores only. It cannot alter route feasibility checks, reduced-cost arithmetic, the master model, or the exact convergence certificate. Malformed checkpoints are rejected by NumPy shape operations but should still be treated as untrusted binary-like data and loaded only from known sources.

## Reporting vulnerabilities

Please report correctness failures, false convergence, route infeasibility, or data leakage through a private maintainer channel before public disclosure.
