#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
from typing import Literal

from datasets import Dataset
from tqdm import tqdm

from sal.config import Config
from sal.utils.math import (
    compute_maj_pred,
    compute_naive_pred,
    compute_weighted_pred,
    extract_completion_answers,
    subsample_completions,
)


def aggregate_scores(
    scores: list[float], agg_strategy: Literal["min", "prod", "last"]
) -> float:
    if agg_strategy == "min":
        return min(scores)
    elif agg_strategy == "prod":
        return math.prod(scores)
    elif agg_strategy == "last":
        return scores[-1]
    else:
        raise ValueError(f"Invalid aggregation strategy: {agg_strategy}")


def score(dataset: Dataset, config: Config) -> Dataset:
    dataset = dataset.map(
        lambda x: {"agg_scores": [aggregate_scores(s, "last") for s in x["scores"]]}
    )
    subsets = [2**i for i in range(config.n) if 2**i <= config.n]
    for n in tqdm(subsets, desc="Computing majority & weighted predictions"):
        dataset = dataset.map(
            subsample_completions,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Subsample {n}",
        )
        dataset = dataset.map(
            extract_completion_answers,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Extract answers {n}",
        )
        dataset = dataset.map(
            compute_weighted_pred,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Compute weighted pred {n}",
        )
        dataset = dataset.map(
            compute_maj_pred,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Compute majority pred {n}",
        )
        dataset = dataset.map(
            compute_naive_pred,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Compute naive pred {n}",
        )
        # Nuke unused columns to keep dataset lean
        dataset = dataset.remove_columns(
            [f"completions@{n}", f"agg_scores@{n}", f"preds@{n}"]
        )
    return dataset
