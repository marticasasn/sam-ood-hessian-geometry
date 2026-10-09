# References

Papers this project builds on, plus the code and datasets it uses.

## Papers

| Year | Title | Authors | Link |
|---|---|---|---|
| 2016 | Deep Residual Learning for Image Recognition | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun | [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |
| 2017 | On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima | Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, Ping Tak Peter Tang | [arXiv:1609.04836](https://arxiv.org/abs/1609.04836) |
| 2019 | Benchmarking Neural Network Robustness to Common Corruptions and Perturbations | Dan Hendrycks, Thomas Dietterich | [arXiv:1903.12261](https://arxiv.org/abs/1903.12261) |
| 2019 | An Investigation into Neural Net Optimization via Hessian Eigenvalue Density | Behrooz Ghorbani, Shankar Krishnan, Ying Xiao | [arXiv:1901.10159](https://arxiv.org/abs/1901.10159) |
| 2020 | PyHessian: Neural Networks Through the Lens of the Hessian | Zhewei Yao, Amir Gholami, Kurt Keutzer, Michael W. Mahoney | [arXiv:1912.07145](https://arxiv.org/abs/1912.07145) |
| 2021 | Sharpness-Aware Minimization for Efficiently Improving Generalization | Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur | [arXiv:2010.01412](https://arxiv.org/abs/2010.01412) |
| 2022 | Towards Understanding Sharpness-Aware Minimization | Maksym Andriushchenko, Nicolas Flammarion | [arXiv:2206.06232](https://arxiv.org/abs/2206.06232) |
| 2023 | How Does Sharpness-Aware Minimization Minimize Sharpness? | Kaiyue Wen, Tengyu Ma, Zhiyuan Li | [arXiv:2211.05729](https://arxiv.org/abs/2211.05729) |
| 2024 | Smoothness-Adaptive Sharpness-Aware Minimization for Finding Flatter Minima | Hiroki Naganuma, Junhyung Lyle Kim, Anastasios Kyrillidis, Ioannis Mitliagkas | Link to verify |

## Code

- **SAM optimizer** (`src/sam.py`) adapted from [davda54/sam](https://github.com/davda54/sam).
- **Hessian estimates** (`src/hessian.py`) computed with [PyHessian](https://github.com/amirgholami/PyHessian).

## Datasets

- **CIFAR-10**: Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009. [Dataset page](https://www.cs.toronto.edu/~kriz/cifar.html)
- **CIFAR-10-C**: Hendrycks & Dietterich (2019), see above. [Zenodo](https://zenodo.org/record/2535967)
