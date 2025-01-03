# Recipes

Here we include YAML configs to run the three test time compute variants detailed in the blog post:

- Best of N: [`recipes/Llama-3.2-1B-Instruct/best_of_n.yaml`](Llama-3.2-1B-Instruct/best_of_n.yaml)
- Beam Search: [`recipes/Llama-3.2-1B-Instruct/beam_search.yaml`](Llama-3.2-1B-Instruct/beam_search.yaml)
- Diverse Verifier Beam Search (DVTS): [`recipes/Llama-3.2-1B-Instruct/dvts.yaml`](Llama-3.2-1B-Instruct/dvts.yaml)

Each approach can be launched by specifying the associated YAML file:

```
python scripts/test_time_compute.py <YAML_CONFIG>
# for example:
python scripts/test_time_compute.py recipes/Llama-3.2-1B-Instruct/best_of_n.yaml
```


The configs shown here are for the `Llama-3.2-1B-Instruct` model, you can override the choice of model by including it in the command line arguments:

```shell
python scripts/test_time_compute.py recipes/Llama-3.2-1B-Instruct/best_of_n.yaml --model_path=Llama-3.2-3B-Instruct --hub_dataset_id=<YOUR_ORG>/Llama-3.2-3B-Instruct-bon-completions
```

> [!WARNING] 
> __best of n__ and __DVTS__ can be run at `n=256` and then subsampled for get comparable solutions for running at `n=4,16,64` etc. The beam search variant **must** be run at the correct `n` in order to make a valid comparison.


## Reproducing results on the MATH-500 dataset

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

The full dataset can then be recontructed with:

```shell
python scripts/merge_chunks.py --dataset_name=<YOUR_ORG>/Llama-3.2-1B-Instruct-bon-completions
```

## Exacting the MATH-500 accuracy numbers

To get the final numbers for the evalations, we use the [Qwen2.5-Math evaluation repo](https://github.com/QwenLM/Qwen2.5-Math), their codebase is well documented, so please refer to their instuctions.


