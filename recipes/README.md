# Recipes

We provide YAML configs to run the three test time compute variants detailed in the [blog post](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute):

| Model | Method |
| :--- | :--- |
| Llama-3.2-1B-Instruct | [Best-of-N](Llama-3.2-1B-Instruct/best_of_n.yaml) |
| | [Beam search](Llama-3.2-1B-Instruct/beam_search.yaml) |
| | [DVTS](Llama-3.2-1B-Instruct/dvts.yaml) |
| Llama-3.2-3B-Instruct | [Best-of-N](Llama-3.2-3B-Instruct/best_of_n.yaml) |
| | [Beam search](Llama-3.2-3B-Instruct/beam_search.yaml) |
| | [DVTS](Llama-3.2-3B-Instruct/dvts.yaml) |
| Qwen2.5-1.5B-Instruct | [Best-of-N](Qwen2.5-1.5B-Instruct/best_of_n.yaml) |
| | [Beam search](Qwen2.5-1.5B-Instruct/beam_search.yaml) |
| | [DVTS](Qwen2.5-1.5B-Instruct/dvts.yaml) |

Each approach can be launched by specifying the associated YAML file, for example:

```shell
export CONFIG=recipes/Llama-3.2-1B-Instruct/best_of_n.yaml

python scripts/test_time_compute.py $CONFIG
```

> [!NOTE]
> For fast testing, each config will generate `n=4` completions over the first 10 problems of the [MATH-500 dataset](https://huggingface.co/datasets/HuggingFaceH4/MATH-500). See below for instruction to replicate the results from our [blog post](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute).

By default, this will save the completions locally to `data/{MODEL_PATH}/{APPROACH}.jsonl`. To push the results as a Hub dataset (recommended), run:

```shell
export CONFIG=recipes/Llama-3.2-1B-Instruct/best_of_n.yaml

python scripts/test_time_compute.py $CONFIG --push_to_hub=true
```

This will push the completions as a _branch_ on the dataset repo; see [here](https://huggingface.co/datasets/lewtun/Llama-3.2-1B-Instruct-best_of_n-prm-completions/tree/HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-4--seed-0--agg_strategy-last) for an example. To convert the branch into a config that can be used for exploration with the dataset viewer, run:

```python
from datasets import load_dataset

DATASET_ID="lewtun/Llama-3.2-1B-Instruct-best_of_n-prm-completions"
REVISION="HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-4--seed-0--agg_strategy-last"

ds = load_dataset(DATASET_ID, revision=REVISION)
# Push to Hub as a config for exploration
ds.push_to_hub(DATASET_ID, config_name=REVISION)
```

To override the choice of model, include it in the command line arguments as follows:

```shell
# Define variables
export CONFIG=recipes/Llama-3.2-1B-Instruct/best_of_n.yaml
export MODEL=meta-llama/Llama-3.2-8B-Instruct

# Run test-time compute
python scripts/test_time_compute.py $CONFIG --model_path=$MODEL
```

> [!WARNING]
> By default, each config will use a chat template that we hand-crafted for Llama 3 models (see the blog post for why). For models that don't use the LLama 3 chat template, set `--custom_chat_template=none`.

Similarly, you can change the choice of dataset (provided it has `problem` and `answer` columns):

```shell
# Define variables
export CONFIG=recipes/Llama-3.2-1B-Instruct/best_of_n.yaml
export DATASET=AI-MO/aimo-validation-aime

# Run test-time compute
python scripts/test_time_compute.py $CONFIG \
    --dataset_name=$DATASET \
    --dataset_split=train
```

Moreover, to override the choice of PRM, include it in the command line arguments as follows:

```shell
# Define variables
export CONFIG=recipes/Qwen2.5-1.5B-Instruct/best_of_n.yaml
export PRM=Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B

# Run test-time compute
python scripts/test_time_compute.py $CONFIG --prm_path=$PRM
```

> Currently supported PRMs: <br>
`RLHFlow/Llama3.1-8B-PRM-Deepseek-Data` (default) <br>
`peiyi9979/math-shepherd-mistral-7b-prm`<br>
`Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B`<br>
`Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B`

## Replicating the blog post results

To replicate the results from our blog post, there are two main steps:

1. Generate completions for each search method and various compute budgets `n`
2. Compute the accuracy of each method for a given compute budget.

Below we provide instructions on how to accomplish each step.

All our experiments were run on H100s with 80GB of VRAM, so you may need to adjust the `--prm_batch_size` and `--search_batch_size` arguments to fit the models on different hardware.

### Generate completions

Below are commands to generate completions for each method. Note that we ran each method across 5 independent seeds to quantify the variability 

**Best-of-N**

> [!NOTE]
> Best-of-N and DVTS only require a single run at `n=256` since the resulting completions can be subsampled for get comparable solutions for running at `n=4,16,64` etc.

```shell
export CONFIG=recipes/Llama-3.2-1B-Instruct/best_of_n.yaml
# Repeat for seeds 0-4
export SEED=0 

python scripts/test_time_compute.py $CONFIG \
    --n=256 \
    --num_samples=500 \
    --seed=$SEED
```

The result will be a dataset like [this](https://huggingface.co/datasets/HuggingFaceH4/Llama-3.2-1B-Instruct-best-of-N-completions)

**Beam search**

Unlike Best-of-N or DVTS which only require a single run at `n=256`, the beam search completions must be generated separately for each value of `n`:

```shell
export CONFIG=recipes/Llama-3.2-1B-Instruct/beam_search.yaml
# Repeat for seeds 0-4
export SEED=0 

for n in 4 16 64 256; do
    python scripts/test_time_compute.py $CONFIG \
        --n=$n \
        --num_samples=500 \
        --seed=$SEED
done
```

The result will be a dataset like [this](https://huggingface.co/datasets/HuggingFaceH4/Llama-3.2-1B-Instruct-beam-search-completions)

**DVTS**

```shell
export CONFIG=recipes/Llama-3.2-1B-Instruct/dvts.yaml
# Repeat for seeds 0-4
export SEED=0 

python scripts/test_time_compute.py $CONFIG \
    --n=256 \
    --num_samples=500 \
    --seed=$SEED
```

The result will be a dataset like [this](https://huggingface.co/datasets/HuggingFaceH4/Llama-3.2-1B-Instruct-DVTS-completions)

#### Speeding up generation via parallelisation

In practice, running each method over the full 500 problems with `n=256` completions is _very slow_ on a single GPU (~3 hours for Best-of-N and ~60+ hours for beam search and DVTS). To speed things up, we provide Slurm scripts that configure array jobs to parallelize the evaluation of the three methods:

```shell
# Best of N
sbatch recipes/launch_array.slurm recipes/Llama-3.2-1B-Instruct/best_of_n.yaml \
    --n=256 \
    --seed=0 \
    --hub_dataset_id <YOUR_ORG>/Llama-3.2-1B-Instruct-best_of_n-completions

# Beamsearch (repeat for n=4,16,64,256)
sbatch recipes/launch_array.slurm recipes/Llama-3.2-1B-Instruct/beam_search.yaml \
    --n=4 \
    --seed=0 \
    --hub_dataset_id=<YOUR_ORG>/Llama-3.2-1B-Instruct-beam_search-completions

# DVTS
sbatch recipes/launch_array.slurm recipes/Llama-3.2-1B-Instruct/dvts.yaml \
    --n 256 \
    --seed 0 \
    --hub_dataset_id=<YOUR_ORG>/Llama-3.2-1B-Instruct-dvts-completions
```

By default this will shard the dataset into 20 chunks of 25 problems each in order to run each algorithm in parallel. The dataset chunks will then be pushed to the Hugging Face Hub as separate branches/revisions.

The full dataset can then be reconstructed from the chunks with:

```shell
python scripts/merge_chunks.py \
    --dataset_name=<YOUR_ORG>/Llama-3.2-1B-Instruct-best_of_n-completions \
    --filter_strings seed-0 # Adjust for each seed
```

## Extracting the MATH-500 accuracy numbers

To get the final numbers for the evalations, we use a [fork](https://github.com/huggingface/Qwen2.5-Math) of the [Qwen2.5-Math evaluation repo](https://github.com/QwenLM/Qwen2.5-Math). Please follow the installation and usage instructions in our fork to obtain accuracies on MATH-500.

> [!NOTE]
> We are working on switching the Qwen-Math parser for an improved one in `lighteval`. Once we have validated the results, we will be able to have a stand-alone evaluation script directly in `search-and-learn`: stay tuned!

## Training Process Reward Models

The [`training` README](training/README.md) is a guide to both train PRMs using [TRL](https://github.com/huggingface/trl) and evaluate them using the [ProcessBench](https://arxiv.org/abs/2412.06559) benchmark, with the code used to fine-tune [Qwen/Qwen2.5-Math-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct) and [Qwen/Qwen2.5-Math-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct) and evaluate the final models.
