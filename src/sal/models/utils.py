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

from dataclasses import dataclass
from functools import cached_property

@dataclass
class Example:
    problem: str
    steps: list[str]
    sep: str = "\n"

    @cached_property
    def get_texts(self):
        """Returns the lists with each problem and solution steps concatenated
        with the separator. 
        """
        return [
            self.sep.join((self.problem, *self.steps[:i])) + self.sep
            for i, step in enumerate(self.steps, start=1)
        ]


class BatchProcessor:
    """Helper class to allow passing batches to the model pipeline including different
    problem and solutions steps. It allows assigning back the steps of the errors at the
    end by finding the corresponding index of the problems in the batches.
    """
    def __init__(self, data: list[Example], batch_size: int = 32):
        self.data = data
        self.batch_size = batch_size
        self.current_idx = 0

        # Create index mapping for steps
        self.step_mapping = []  # [(dataset_idx, step_idx), ...]
        for idx, item in enumerate(data):
            for step_idx in range(len(item.steps)):
                self.step_mapping.append((idx, step_idx))

        self.total_steps = len(self.step_mapping)

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.total_steps:
            raise StopIteration

        batch_indices = []
        batch_steps = []
        step_count = 0

        while self.current_idx < self.total_steps and step_count < self.batch_size:
            dataset_idx, step_idx = self.step_mapping[self.current_idx]
            batch_indices.append((dataset_idx, step_idx))

            # Here the steps have to be already generated
            steps = self.data[dataset_idx].get_texts
            batch_steps.append(steps[step_idx])

            step_count += 1
            self.current_idx += 1

        return batch_steps, batch_indices

    def get_total_batches(self):
        """Return the total number of batches."""
        return (self.total_steps + self.batch_size - 1) // self.batch_size


def process_results(
    results: list[dict[str, bool | str | int]],
    batch_indices: list[tuple[int, int]],
    processed_data: dict[int, list[dict[str, str | float | int]]]
) -> None:
    """
    Assign results back to the original dataset structure.

    Args:
        results: List of results from processing the batch,
            the outputs from transformers.pipeline(X).
        batch_indices: List of (dataset_idx, step_idx) tuples.
        processed_data: Dictionary to store results, keyed by dataset index.
    """
    for result, (dataset_idx, step_idx) in zip(results, batch_indices):
        if dataset_idx not in processed_data:
            processed_data[dataset_idx] = []
        # Ensure the list is long enough to insert at step_idx
        while len(processed_data[dataset_idx]) <= step_idx:
            processed_data[dataset_idx].append(None)
        processed_data[dataset_idx][step_idx] = result
