# Exactness and Safety Contract

## What is exact

The full pricing routine solves the declared elementary shortest-path problem with capacity and hard time-window resources by label setting. Labels with the same visited set and terminal node are compared on departure time and open reduced cost. A label is removed only when another label is no later and no more expensive, which preserves all potentially improving continuations.

For small instances, regression tests compare the resulting minimum reduced cost with complete feasible-route enumeration.

The route-based master is a linear set-partitioning relaxation. When full pricing proves that no route has reduced cost below the configured tolerance, the restricted-master solution is optimal for the full route-based LP over the declared route universe.

## Why hybrid pricing remains correct

Restricted pricing is a heuristic oracle. It may omit the globally best route or every negative-reduced-cost route. Its output is used only as follows:

1. if it finds a valid negative-reduced-cost route, that route is added;
2. if it does not, full exact pricing is invoked;
3. the algorithm terminates only after full pricing returns no negative route.

Column generation does not require the most negative column at every iteration. Any negative-reduced-cost column makes valid progress. The final full-pricing call supplies the convergence certificate.

## What is not exact

The binary master over generated columns is a restricted integer problem. Root-node column generation alone does not prove the global integer optimum, because a route with nonnegative reduced cost for the final LP may still be needed in an optimal integer solution.

For tiny instances the benchmark can enumerate every feasible route and solve the complete integer master. Only that finite oracle is reported as a full integer optimum.

The repository does not implement branch-and-price, branching-compatible pricing, cuts, dual stabilization, or ng-route relaxations.
