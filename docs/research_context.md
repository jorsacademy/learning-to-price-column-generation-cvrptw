# Research Context

Column generation repeatedly solves a restricted master and a pricing subproblem. In routing applications, pricing commonly becomes an elementary shortest-path problem with resource constraints, and its label-setting algorithm can dominate runtime.

Morabit, Desaulniers, and Lodi study machine-learning-based arc selection for constrained-shortest-path pricing. Their central idea is to predict arcs likely to matter and solve a smaller pricing network. This repository adopts that broad design pattern, but it is an independent compact implementation with synthetic data, a NumPy MLP, and a mandatory exact fallback.

Abouelrous et al. study an attention-based reinforcement-learning mechanism that independently constructs pricing columns for a vehicle-routing application. That is a different and more ambitious learning target. This repository does not reproduce that architecture; end-to-end policy pricing is left as a future extension.

Modern CVRPTW column-generation research also uses stronger non-ML machinery, including local-area routes, graph-master representations, ng-route relaxations, bidirectional labeling, completion bounds, dual stabilization, valid inequalities, and branch-price-and-cut. Learned pricing should be evaluated against such algorithmic improvements rather than presented as a substitute for them.

Primary references:

- https://doi.org/10.1287/ijoo.2022.0082
- https://arxiv.org/abs/2504.02383
- https://arxiv.org/abs/2304.11723
- https://doi.org/10.1287/opre.1070.0449
