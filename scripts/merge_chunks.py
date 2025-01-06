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
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from typing import List

from datasets import concatenate_datasets, load_dataset
from tqdm.auto import tqdm
from transformers import HfArgumentParser

from sal.utils.hub import get_dataset_revisions

"""Merge revisions of a dataset into a single config.

Usage:

# Merge all revisions of a dataset for a given seed
python scripts/merge_chunks.py \
    --dataset_name HuggingFaceH4/Llama-3.2-1B-Instruct-best-of-N-completions \
    --filter_strings seed-0

# Merge only revisions that contain "last" or "T-0.0" or "seed-0" in their name
python scripts/merge_chunks.py \
    --dataset_name HuggingFaceH4/Llama-3.2-1B-Instruct-best-of-N-completions \
    --filter_strings last T-0.0 seed-0
"""


@dataclass
class Args:
    dataset_name: str
    dataset_split: str = "train"
    filter_strings: List[str] = field(default_factory=list)
    hub_dataset_private: bool = False


def load_single_revision(args):
    """Load a single dataset revision."""
    dataset_name, revision, dataset_split = args
    return load_dataset(
        dataset_name,
        revision=revision,
        trust_remote_code=True,
        split=dataset_split,
        download_mode="force_redownload",
    )


def main():
    parser = HfArgumentParser(Args)
    args = parser.parse_args_into_dataclasses()[0]
    revisions = get_dataset_revisions(args.dataset_name)

    if args.filter_strings:
        revisions = [
            revision
            for revision in revisions
            if all(filter_string in revision for filter_string in args.filter_strings)
        ]

    merged_config = revisions[0].split("--chunk")[0]
    print(f"Merging {len(revisions)} revisions to create config `{merged_config}`")

    # Prepare arguments for multiprocessing
    pool_args = [
        (args.dataset_name, revision, args.dataset_split) for revision in revisions
    ]

    # Use multiprocessing to load datasets in parallel
    with Pool(cpu_count()) as pool:
        datasets = list(
            tqdm(
                pool.imap(load_single_revision, pool_args),
                total=len(revisions),
                desc="Loading datasets",
            )
        )

    # Concatenate datasets
    merged_dataset = concatenate_datasets(datasets)

    # Sanity check
    if "problem" in merged_dataset.column_names and len(
        merged_dataset.unique("problem")
    ) != len(merged_dataset):
        raise ValueError("Found duplicate problems")
    if "lighteval_MATH" in merged_config and len(merged_dataset) != 5000:
        raise ValueError(f"Expected 5000 samples, got {len(merged_dataset)}")
    if "MATH-500" in merged_config and len(merged_dataset) != 500:
        raise ValueError(f"Expected 500 samples, got {len(merged_dataset)}")

    # Push merged dataset to the hub
    url = merged_dataset.push_to_hub(
        args.dataset_name,
        config_name=merged_config,
        split=args.dataset_split,
        private=args.hub_dataset_private,
    )
    print(f"Pushed merged dataset to {url}")


if __name__ == "__main__":
    main()
