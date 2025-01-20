<p align="center">
  <img style="width:200px" src="https://raw.githubusercontent.com/huggingface/search-and-learn/main/assets/logo.png">
</p>

<p align="center">
      🤗 <a href="https://huggingface.co/collections/HuggingFaceH4/scaling-test-time-compute-with-open-models-675c3b475a0d6eb4528fec23" target="_blank">Models & Datasets</a> |
      📃 <a href="https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute" target="_blank">Blog Post</a>
</p>

# Search and Learn

Recipes to enhance LLM capabilities by scaling inference-time compute. Name inspired by Rich Sutton's [Bitter Lesson](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf):

> One thing that should be learned from the bitter lesson is the great power of general purpose methods, of methods that continue to scale with increased computation even as the available computation becomes very great. The two methods that seem to scale arbitrarily in this way are _**search**_ and _**learning**_.

## What is this?

Over the last few years, the scaling of _**train-time compute**_ has dominated the progress of LLMs. Although this paradigm has proven to be remarkably effective, the resources needed to pretrain ever larger models are becoming prohibitively expensive, with billion-dollar clusters already on the horizon. This trend has sparked significant interest in a complementary approach: _**test-time compute scaling.**_ Rather than relying on ever-larger pretraining budgets, test-time methods use dynamic inference strategies that allow models to “think longer” on harder problems. A prominent example is OpenAI’s o1 model, which shows consistent improvement on difficult math and coding problems as one increases the amount of test-time compute.

Although we don't know how o1 was trained, Search and Learn aims to fill that gap by providing the community with a series of recipes that enable open models to solve complex problems if you give them enough “time to think”. 

## News 🗞️

* **December 16, 2024**: Initial release with code to replicate the test-time compute scaling results of our [blog post](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute).

## How to navigate this project 🧭

This project is simple by design and mostly consists of:

* [`scripts`](./scripts/) to scale test-time compute for open models. 
* [`recipes`](./recipes/) to apply different search algorithms at test-time. Three algorithms are currently supported: Best-of-N, beam search, and Diverse Verifier Tree Search (DVTS). Each recipe takes the form of a YAML file which contains all the parameters associated with a single inference run. 

To get started, we recommend the following:

1. Follow the [installation instructions](#installation-instructions) to set up your environment etc.
2. Replicate our test-time compute results by following the [recipe instructions](./recipes/README.md).

## Contents

The initial release of Search and Learn will focus on the following techniques:

* **Search against verifiers:** guide LLMs to search for solutions to "verifiable problems" (math, code) by using a stepwise or process reward model to score each step. Includes techniques like Best-of-N sampling and tree search.
* **Training process reward models:** train reward models to provide a sequence of scores, one for each step of the reasoning process. This ability to provide fine-grained feedback makes PRMs a natural fit for search methods with LLMs.


# Installation instructions

To run the code in this project, first, create a Python virtual environment using e.g. Conda:

```shell
conda create -n sal python=3.11 && conda activate sal
```

```shell
pip install -e '.[dev]'
```

Next, log into your Hugging Face account as follows:

```shell
huggingface-cli login
```

Finally, install Git LFS so that you can push models to the Hugging Face Hub:

```shell
sudo apt-get install git-lfs
```

You can now check out the `scripts` and `recipes` directories for instructions on how to scale test-time compute for open models!

## Project structure

```
├── LICENSE
├── Makefile                    <- Makefile with commands like `make style`
├── README.md                   <- The top-level README for developers using this project
├── recipes                     <- Recipe configs, accelerate configs, slurm scripts
├── scripts                     <- Scripts to scale test-time compute for models
├── pyproject.toml              <- Installation config (mostly used for configuring code quality & tests)
├── setup.py                    <- Makes project pip installable (pip install -e .) so `sal` can be imported
├── src                         <- Source code for use in this project
└── tests                       <- Unit tests
```

## Replicating our test-time compute results

The [`recipes` README](recipes/README.md) includes launch commands and config files in order to replicate our results.


## Citation

If you find the content of this repo useful in your work, please cite it as follows via `\usepackage{biblatex}`:

```
@misc{beeching2024scalingtesttimecompute,
      title={Scaling test-time compute with open models},
      author={Edward Beeching and Lewis Tunstall and Sasha Rush},
      url={https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute},
}
```

Please also cite the original work by DeepMind upon which this repo is based:

```
@misc{snell2024scalingllmtesttimecompute,
      title={Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters}, 
      author={Charlie Snell and Jaehoon Lee and Kelvin Xu and Aviral Kumar},
      year={2024},
      eprint={2408.03314},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.03314}, 
}
```

