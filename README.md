# Search and Learn

> One thing that should be learned from [the bitter lesson](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf) is the great power of general purpose methods, of methods that continue to scale with increased computation even as the available computation becomes very great. The two methods that seem to scale arbitrarily in this way are _**search**_ and _**learning**_.

# Installation

```shell
conda create -n sal python=3.10 && conda activate sal
```
```shell
pip install -e '.[dev]'
```


## Replicating Scaling Test Time Compute results:
The [recipes readme](recipes/README.md) includes launch commands and config files in order to replicate our results.


## Citation
If you use this codebase or the blogpost it would be great if you could cite us:
```
@misc{beeching2024scalingtesttimecompute,
      title={Scaling test-time compute with open models},
      author={Edward Beeching and Lewis Tunstall and Sasha Rush},
      url={https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute},
}
```
Please also cite this work:
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

