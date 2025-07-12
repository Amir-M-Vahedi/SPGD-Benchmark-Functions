---
title: "SPGD: Steepest Perturbed Gradient Descent"
layout: default
---

# Steepest Perturbed Gradient Descent (SPGD)

**Authors**: Amir M. Vahedi, Horea T. Ilies  
📄 **Paper**: [arXiv:2411.04946](https://arxiv.org/abs/2411.04946)

---

## 🚀 Overview

The Steepest Perturbed Gradient Descent (SPGD) Algorithm introduces a novel optimization approach by integrating periodic perturbations into the gradient descent process. It enhances the ability to escape local minima, handle saddle points, and traverse flat regions in non-convex optimization problems.

<p align="center">
  <img width="900" src="https://github.com/user-attachments/assets/c0253dc9-06db-41eb-b2be-2fe601287367" alt="SPGD Pseudocode">
</p>

SPGD is especially effective in high-dimensional and rugged objective landscapes.

---

## 🔔 What's New

- ✅ **Now on arXiv**: [https://arxiv.org/abs/2411.04946](https://arxiv.org/abs/2411.04946)
- ✅ **Bayesian Optimization** added as a comparison method
- ✅ **30 random-start trials** introduced to evaluate robustness

---

## ⚔️ Compared Methods

- Gradient Descent (GD)  
- Perturbed Gradient Descent (PGD)  
- Simulated Annealing (SA)  
- Fminunc (MATLAB)  
- Fmincon (MATLAB)  
- **Bayesian Optimization** ✅

SPGD consistently showed strong performance and robustness compared to other baselines.

---

## 📊 Benchmark Functions

We evaluated on several 2D benchmark functions:

- **Peaks Function** – Multiple local minima & flat regions  
- **Ackley Function** – Deceptively complex surface  
- **Easom Function** – Sharp global minimum in a flat region  
- **Lévi N.13 Function** – Highly oscillatory surface  
- **Rosenbrock Function** – Long narrow valley  
- **Drop-Wave Function** – Multiple sharp ripples

Each of these was chosen to test SPGD's ability to escape traps and locate the global optimum.

---

## 📖 Citation

```bibtex
@misc{SPGD,
      title={SPGD: Steepest Perturbed Gradient Descent Optimization}, 
      author={Amir M. Vahedi and Horea T. Ilies},
      year={2024},
      eprint={2411.04946},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2411.04946}, 
}
