# Steepest Perturbed Gradient Descent (SPGD) Algorithm

## Overview
The Steepest Perturbed Gradient Descent (SPGD) Algorithm introduces a novel approach to optimization by integrating periodic perturbations into the gradient descent process. This method enhances the capability to escape local minima and navigate complex optimization landscapes more effectively. It is particularly well-suited for dealing with non-convex functions and those with challenging topologies such as multiple local minima, high-dimensional spaces, and saddle points. The detailed steps of the algorithm are shown below:
<p align="center">
  <img width="1000" alt="Screenshot 2022-10-07 220752 - Copy" src="https://github.com/user-attachments/assets/c0253dc9-06db-41eb-b2be-2fe601287367">
</p>
This pseudocode illustrates how SPGD periodically introduces perturbations at predefined intervals (controlled by `Iter_p`), evaluates these perturbations, and potentially shifts the search trajectory towards more promising regions in the solution space.

## Comparison with Other Optimization Methods
To highlight the capabilities and advantages of the SPGD algorithm, we compare its performance against several established optimization techniques:
- **Gradient Descent (GD)**: A traditional approach often used due to its simplicity and effectiveness in convex problems.
- **Perturbed Gradient Descent (PGD)**: Enhances GD by incorporating random perturbations to escape local minima potentially after getting trapped in one.
- **MATLAB Simulated Annealing (SA)**: A probabilistic technique that attempts to avoid getting stuck in local optimum by allowing hill-climbing moves depending on a temperature parameter that decreases over time.
- **Fmincon (MATLAB)**: A versatile MATLAB function that handles various types of constraints and is capable of finding constrained minima. These comparisons are integral to demonstrating SPGD's robustness and versatility across different problem domains.

## Benchmark Functions
This repository contains implementations of several 2D benchmark functions used to evaluate the effectiveness of the SPGD algorithm. Below are the functions and their descriptions, challenges, and mathematical formulas:

### 1. Peaks Function
- **Formula:** `PEAKS`
 
$$f(x, y) = 3(1-x)^2 e^{-(x^2) - (y+1)^2} - 10(\frac{x}{5} - x^3 - y^5) e^{-x^2-y^2} - \frac{1}{3} e^{-(x+1)^2 - y^2}$$

- **Challenge:** features a single global minimum, numerous local minima, a saddle point, and extensive flat regions, challenging the algorithms to differentiate and navigate these varied features effectively.
<p align="center">
  <img width="500" alt="Screenshot 2022-10-07 220752 - Copy" src="https://github.com/user-attachments/assets/d713dadb-0d7e-4bb2-8105-b63c47eeda44">
</p>

### 2. Ackley Function
- **Formula:** `Ackley`

$$f(x, y) = -20 \exp\left(-0.2 \sqrt{0.5(x^2 + y^2)}\right) - \exp\left(0.5 (\cos(2\pi x) + \cos(2\pi y))\right) + e + 20$$

- **Challenge:** Known for its deceptive appearance. It features a global optimum surrounded by a multitude of local minima, requiring algorithms to overcome local attractions and efficiently search for the global optimum across a complex, multidimensional space.
<p align="center">
  <img width="500" alt="Screenshot 2022-10-07 220752 - Copy" src="https://github.com/user-attachments/assets/37583d9e-636e-4b99-b458-08411829bf12">
</p>

### 3. Easom Function
- **Formula:** `Easom`
  
$$f(x, y) = -\cos(x) \cos(y) \exp(-(x-\pi)^2-(y-\pi)^2)$$

- **Challenge:** Characterized by a sharp peak at the global minimum surrounded by a flat region. The function is particularly challenging due to the difficulty in locating the peak in a large search space.
<p align="center">
  <img width="500" alt="Screenshot 2022-10-07 220752 - Copy" src="https://github.com/user-attachments/assets/39bf2b7d-242d-40d9-99e7-4a4f0d6faf81">
</p>

### 4. Lévi Function N.13
- **Formula:** `Lévi N.13`

$$f(x, y) = \sin^2(3\pi x) + (x-1)^2(1+\sin^2(3\pi y)) + (y-1)^2(1+\sin^2(2\pi y))$$

- **Challenge:** Features a complex, wavy surface with many local minima, which makes finding the global minimum challenging without sophisticated exploration strategies.
<p align="center">
  <img width="500" alt="Screenshot 2022-10-07 220752 - Copy" src="https://github.com/user-attachments/assets/99bf4d11-a935-4c8b-b0a1-29e0b4e7812f">
</p>

### 5. Rosenbrock Function
- **Formula:** `Rosenbrock`
  
$$f(x, y) = (a - x)^2 + b(y - x^2)^2$$

where typically $\( a = 1 \)$ and $\( b = 100 \)$.
- **Challenge:** Known as the valley or banana function, it contains a narrow, parabolic shaped flat valley. Finding the valley and then the minimum is computationally challenging.
<p align="center">
  <img width="700" alt="Screenshot 2022-10-07 220752 - Copy" src="https://github.com/user-attachments/assets/86ad6619-5227-40f0-aaf8-66ebb3ce3d27">
</p>

### 6. Drop-Wave Function
- **Formula:** `DROP-WAVE`

$$f(x, y) = -\frac{1+\cos(12\sqrt{x^2+y^2})}{0.5(x^2+y^2)+2}$$
  
- **Challenge:** The function has multiple ripples that cause numerous local minima, complicating the path to the global minimum.
<p align="center">
  <img width="700" alt="Screenshot 2022-10-07 220752 - Copy" src="https://github.com/user-attachments/assets/8fc69efe-2d45-44dc-b87c-093b41c0f57f">
</p>

## Citation
Please cite the following paper if you use this algorithm in your research:
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
```
## Contributions
Contributions are welcome! If you have any suggestions or improvements, please fork this repository and open a pull request, or create an issue with your recommendations.
