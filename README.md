# Nature-inspired Optimization Algorithms: A Comparative Study

nature_inspired_optimization_algorithms_comperative/
│
├── requirements.txt                    # Project dependencies
├── main_simulation.py                  # Entry point for the 1530 statistical runs
├── config.py                           # Global configurations (hyperparameters, FEs, bounds)
│
├── src/                                # Source code directory
│   ├── __init__.py
│   │
│   ├── algorithms/                     # Meta-heuristic algorithm implementations
│   │   ├── __init__.py
│   │   ├── base_optimizer.py           # Abstract base class (OOP inheritance)
│   │   ├── pso.py                      # Particle Swarm Optimization
│   │   ├── de.py                       # Differential Evolution
│   │   ├── gwo.py                      # Grey Wolf Optimizer
│   │   ├── abc.py                      # Artificial Bee Colony
│   │   └── es.py                       # Evolution Strategies
│   │
│   ├── benchmarks/                     # Mathematical test environments
│   │   ├── __init__.py
│   │   └── functions.py                # Sphere, Rastrigin, Rosenbrock definitions
│   │
│   └── utils/                          # Helper modules
│       ├── __init__.py
│       ├── logger.py                   # I/O operations (JSON/CSV data logging)
│       ├── statistics_engine.py        # Wilcoxon, Friedman, Mean, Median, Std calculations
│       └── visualizer.py               # Convergence curves, boxplots, and 2D animations
│
│
└── outputs/                            # Generated artifacts
    ├── figures/                        # Static convergence and boxplot images
    ├── statistics/                     # Output of 1530 runs
    └── animations/                     # 3D surface GIFs/MP4s from showcase runs


This repository contains a comprehensive comparative analysis of nature-inspired meta-heuristic optimization algorithms. The study evaluates performance, scalability, and statistical reliability across multi-dimensional benchmark functions.

## 1. Project Overview
The core objective is to rationalize the behavior of stochastic optimizers when facing the **Curse of Dimensionality ($D \in \{10, 20, 30\}$)**. The framework integrates automated simulation pipelines with rigorous statistical validation.

### Supported Algorithms
- **ABC**: Artificial Bee Colony
- **DE**: Differential Evolution
- **ES**: Evolution Strategies
- **GWO**: Grey Wolf Optimizer
- **PSO**: Particle Swarm Optimization

### Benchmark Functions
- **Sphere**: Unimodal, smooth convex function.
- **Ackley**: Multimodal, characterized by a large number of local minima.
- **Zakharov**: Plate-shaped, non-separable function with a flat surface.

## 2. Methodology & Optimization Logic

### Success Criteria & Early Stopping
To ensure computational efficiency, an **Early Stopping** mechanism is implemented. A run is considered a "Success" if the fitness value $f(x)$ reaches the success threshold $\epsilon$:
$$f(x) < \epsilon, \quad \text{where } \epsilon = 10^{-8}$$

### Success Rate (SR) Calculation
The Success Rate represents the reliability of the algorithm over $N = 30$ independent runs:
$$SR = \\left( \\frac{\sum_{i=1}^{N} S_i}{N} \\right) \\times 100$$
where $S_i \in \{0, 1\}$ denotes the success status of the $i$-th run.

## 3. Statistical Framework

### Two-sided Binomial Test
To prove that results are not due to random chance, a **Two-sided Binomial Test** is applied to the Success Rate (SR). We test the deviation from a random baseline ($p_{null} = 0.5$):

- **Null Hypothesis ($H_0$):** The algorithm performs no better/worse than random guessing ($p = 0.5$).
- **Alternative Hypothesis ($H_1$):** The performance is significantly different from random ($p \\neq 0.5$).

The Probability Mass Function (PMF) used for the exact test:
$$P(X = k) = \\binom{n}{k} p^k (1-p)^{n-k}$$
where $n=30$ and $k$ is the number of successes. A result is considered **statistically significant** if the $p\text{-value} < 0.05$.
- **Reliability Index:** Color-coded p-values indicating the deterministic nature of success/failure.
    ├── algorithms/          # Meta-heuristic implementations
    ├── benchmarks/          # Mathematical test functions
    └── utils/               # Statistics engine & visualizer
