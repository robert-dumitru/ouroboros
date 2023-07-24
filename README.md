# Ouroboros

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Ouroboros is a simple autograd engine, built on top of numpy. I originally wrote it as a learning exercise (drawing heavy inspiration from [micrograd](https://github.com/karpathy/micrograd)), but it performs surprisingly well. It's not intended for production use, but it's a fun way to learn about autograd.

## Installation

Clone the repository and install the dependencies with poetry:

```bash
poetry install
```

## Usage

Ouroboros is designed to be used in a similar way to PyTorch. You can create tensors from numpy arrays, and perform operations on them. The operations are recorded, and you can call `.backward()` to compute the gradients of the tensors with respect to the operations. Additionally, for convenience, you can use the `grad` function to compute the gradient of a function given a certain input.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

