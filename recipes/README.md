# Recipes

We provide YAML configs to run the three test time compute variants detailed in the [blog post](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute):

| Model | Method |
| :--- | :--- |
| Llama-3.2-1B-Instruct | [Best-of-N](Llama-3.2-1B-Instruct/best_of_n.yaml) |
| | [Beam search](Llama-3.2-1B-Instruct/beam_search.yaml) |
| | [DVTS](Llama-3.2-1B-Instruct/dvts.yaml) |
| Llama-3.2-3B-Instruct | [Best-of-N](Llama-3.2-1B-Instruct/best_of_n.yaml) |
| | [Beam search](Llama-3.2-3B-Instruct/beam_search.yaml) |
| | [DVTS](Llama-3.2-3B-Instruct/dvts.yaml) |

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

This will push the completions as a _branch_ on the dataset repo; see [here](https://huggingface.co/datasets/lewtun/Llama-3.2-1B-Instruct-best_of_n-prm-completions/tree/HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-4--seed-0--agg_strategy-last) for an example. To load the dataset for further post-processing, run:

```python
from datasets import load_dataset

ds = load_dataset("lewtun/Llama-3.2-1B-Instruct-best_of_n-prm-completions", revision="HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-4--seed-0--agg_strategy-last")
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
> By default, each config will use a chat template that is hand-crafted for Llama 3 models. To use the model's default chat template, set `--custom_chat_template=none`.

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

## Replicating the results from the blog post

> [!WARNING] 
> __best of n__ and __DVTS__ only require a single run at `n=256` since the resulting completions can be subsampled for get comparable solutions for running at `n=4,16,64` etc. To obtain the beam search **must** be run at separate values of `n` in order to make a valid comparison with the other methods at the same `n`.

We provide Slurm scripts to configure array jobs to parallelize the evaluation of the three methods:


```shell
# Best of n
sbatch recipes/launch_array.slurm recipes/Llama-3.2-1B-Instruct/best_of_n.yaml \
    --hub_dataset_id=<YOUR_ORG>/Llama-3.2-1B-Instruct-bon-completions
# Beamsearch n=4,16,64,256
sbatch recipes/launch_array.slurm recipes/Llama-3.2-1B-Instruct/beam_search.yaml --n=4 \
    --hub_dataset_id=<YOUR_ORG>/Llama-3.2-1B-Instruct-beam-search-completions
# DVTS n=16
sbatch recipes/launch_array.slurm recipes/Llama-3.2-1B-Instruct/dvts.yaml --n=16 \
    --hub_dataset_id=<YOUR_ORG>/Llama-3.2-1B-Instruct-dvts-completions
```
By default this will shard the dataset into 20 chunks in order to run the algorithm in parallel, the dataset will be pushed to the Hugging Face hub. 

The full dataset can then be reconstructed with:

```shell
python scripts/merge_chunks.py --dataset_name=<YOUR_ORG>/Llama-3.2-1B-Instruct-bon-completions
```

## Extacting the MATH-500 accuracy numbers

To get the final numbers for the evalations, we use the [Qwen2.5-Math evaluation repo](https://github.com/QwenLM/Qwen2.5-Math), their codebase is well documented, so please refer to their instuctions.


