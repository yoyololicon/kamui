# Kamui

Kamui is a python package for robust and accurate phase unwrapping on 2D, 3D, or sparse data. 

Kamui unwrap the phases by viewing the data points as vertices $`V = \{v: v \in [0, 2\pi)\}`$ connected with edges $`E = \{(u, v): u, v\in V \cap u \neq v\}`$ and solving the following integer linear programming (ILP) problem:

```math
\min_{k} w^T |k|,
```

```math
\text{s.t.} Ak = -A\frac{x}{2\pi},
```
where $`k_{i \in [0, M)} \in \mathbb{Z}`$ is the point ambiguities, $`w_{i \in [0, M)} \in \mathbb{R}^+`$ is the weights, $`x_i = \{(v - u + \pi) \pmod {2\pi} - \pi: (u, v) \in E\}_{i \in [0, M)}`$ is the pseudo phase derivatives, $`M = |E|`$. 
$`A_{ij} \in \{-1, 0, 1\} \cap i \in [0, N), j \in [0, M)`$ and $N$ is the number of elementary cycles enclosed by $E$.

This formulation is based on the fact that the true phase differences, $2\pi k + x$, should fulfill the irrotationality constraint, which means the summation of phase derivatives of each elementary cycles is zeros.
This is the general form of the network programming approach proposed in the paper "[A novel phase unwrapping method based on network programming](https://ieeexplore.ieee.org/document/673674)".

Unwrapping phase with Kamui can be computationally heavy due to the fact that ILP is NP-hard.
Acceleration techniques, such as dividing the graph into subgraphs, will be implemented in the future.

## Installation

```commandline
pip install kamui
```

Kamui also provide [PUMA](https://ieeexplore.ieee.org/document/4099386), a fast and robust phase unwrapping algorithm based on graph cuts as an alternative.
To install PUMA, run

```commandline
pip install kamui[extra]
```

However, it uses the original maxflow implementation by Vladimir Kolmogorov with GPL license.
Please follow the licensing instruction in [PyMaxFlow](http://pmneila.github.io/PyMaxflow/#indices-and-tables) if you use this version of kamui.


## Usage

WIP.

```python
```

## TODO

- [ ] subgraph division


## References

- [A novel phase unwrapping method based on network programming](https://ieeexplore.ieee.org/document/673674)
- [Phase Unwrapping via Graph Cuts](https://ieeexplore.ieee.org/document/4099386)