<div align="center">
  <img src="editax-logo.svg" alt="Editax Logo" width="200">

  <h1>Editax</h1>

  <p>A framework for automated ACCEL-type editors for Unsupervised Environment Design</p>

  <div>
    <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License">
    <a href="https://github.com/yourusername/editax/issues">
      <img src="https://img.shields.io/github/issues/yourusername`/editax" alt="Issues">
    </a>
  </div>
</div>`

## Overview

Editax is a framework for creating automated ACCEL (Evolving Curricula with Regret-Based Environment Design) editors for Unsupervised Environment Design. It automatically generates small probabilistic programs to modify POMDPs, removing the domain knowledge required to hand-design editors.

### Key Features

- 🔄 **Automated Editor Generation**: Creates and validates environment modifications automatically
- 🎯 **Minimal Perturbation**: Ensures changes are meaningful yet minimal
- 🤖 **Consistency Sampling**: Applies universal self-consistency to minimize differences across LLM invocations
- 🔬 **Framework Integration**: Seamlessly integrates with minimax and jaxued for end-to-end training

## Motivation

Implementing ACCEL requires significant domain knowledge for each UPOMDP. Editax streamlines this process by:

- **Automating Editor Creation:** Generates ACCEL-based modifications automatically


## Installation

### Prerequisites
- Python 3.12+
- pip


### Installing from PyPI

If you prefer not to clone the repository, you can install the main package from PyPI:

```bash
# Install the main package
pip install editax

# For full functionality, install the git dependency separately:
pip install git+https://github.com/DramaCow/jaxued.git@main
```

### Development Installation

For development work (including dev dependencies such as `jaxued`), clone the repository and install using Poetry with dev extras:

```bash
git clone https://github.com/RobbenRibery/editax.git
cd editax
python -m venv .venv
source .venv/bin/activate
poetry install --with dev
```

### Environment Setup

Configure your language model API keys:
```bash
export OPENAI_API_KEY="your-key-here"
# Or alternatively:
export ANTHROPIC_API_KEY="your-key-here"
export DEEPSEEK_API_KEY="your-key-here"
```

## Quick Start

Generate environment editors (i.e. for kinetix):

```bash
python -m experiments.gen_editors_kinetix
```

This will:
1. Load your target environment
2. Generate editor functions
3. Validate with a sampled environment and refine the modifications
4. Output ready-to-use environment editors

## Contributing

We welcome contributions! To get started:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

Please read our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.


## References

[1] Parker-Holder, J., et al. (2022). [Evolving Curricula with Regret-Based Environment Design](https://arxiv.org/abs/2203.01302)

[2] Matthews, M., et al. (2024). [Kinetix: Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks](https://arxiv.org/abs/2410.23208)

[3] Coward, S., et al. (2024). [JaxUED: A simple and useable UED library in Jax](https://arxiv.org/abs/2403.13091)

[4] Matthews, M., et al. (2024). [Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning](https://arxiv.org/abs/2402.16801)

[5] Jiang, M., et al. (2023). [Minimax Regret Environment Design for Open-Ended Learning](https://arxiv.org/abs/2311.12716)

[6] Chen, X., et al. (2023). [Universal Self-Consistency for Large Language Model Generation](https://arxiv.org/abs/2311.17311)

## Citation

If you use this software in your research, please cite it as:

```bibtex
@software{editax2024,
  title = {Editax: A Framework for Automated ACCEL-type Editors},
  author = {
    Liu, Rundong and 
    Sudhakaran, Shyam and
    Beck, Jacob and
    Zhang, Richard and
    Anisimov, Maksim
  },
  year = {2024},
  url = {https://github.com/RobbenRibery/editax},
  email = {
    rundong.liu{\@}proton.me and
    shyamsnair{\@}protonmail.com and
    jakeabeck{\@}gmail.com and
    zrich107{\@}gmail.com and
    m.anisimov23{\@}imperial.ac.uk`
  }
}
```
