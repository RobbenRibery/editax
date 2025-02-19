![Editax Logo](editax-logo.svg)

# Editax

Editax is a framework for creating automated ACCEL editors for Unsupervised Environment Design. Its primary goal is to generate and refine modifications to reinforcement learning (RL) environments automatically. By leveraging advanced language models and minimal meaningful perturbation strategies, Editax provides a scalable way to design environments that both ease agents into training and continuously challenge them as they progress.

## Motivation

Designing effective RL environments is crucial for training robust, adaptable agents. Traditional methods require considerable manual effort and iterative fine-tuning, which can slow down research and development. Editax addresses these challenges by:

- **Automating Editor Creation:** Uses ACCEL (Automated Curriculum Component Editor Logic) techniques to automatically generate and correct environment modifications.
- **Accelerating Experimentation:** Enables rapid prototyping of environment changes without manual intervention.
- **Enhancing Unsupervised Design:** Applies unsupervised methods to derive minimal meaningful perturbations that gradually adjust the difficulty of the environment.

This automated approach allows researchers and practitioners to experiment with environment designs more effectively and efficiently.

## Getting Started with Potery

Potery is an integral part of the Editax ecosystem that simplifies the process of generating and integrating environment editors. It streamlines the creation, testing, and correction of editor functions.

### Local Setup

Follow these steps to set up Editax locally:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/editax.git
   ```

2. **Navigate to the Repository Directory:**
   ```bash
   cd editax
   ```

3. **Set Up a Virtual Environment and Install Dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install poetry
   poetry install``
   ```

4. **Set Up Environment Variables:**
   Ensure that API keys for your chosen language model are set (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `DEEPSEEK_API_KEY`).

### Running Potery

To start the automated editor generation with Potery, run:

```bash
python -m experiments.gen_editors_kinetix
```

This command will:
- Load the target RL environment source code.
- Generate candidate editor functions based on the ACCEL approach.
- Iteratively apply corrections and refinements to produce a robust and functioning set of environment modifications.

## Contributing

Contributions to Editax are welcome! If you have suggestions, improvements, or bug fixes, please open an issue or submit a pull request on our GitHub repository.

## License

Editax is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
