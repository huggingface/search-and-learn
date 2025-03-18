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
import random
import signal
from collections import defaultdict
from multiprocessing import Manager
from typing import Any, Dict, List, Literal

import numpy as np
from latex2sympy2 import latex2sympy
from sympy import latex, simplify

from .qwen_math_parser import extract_answer, strip_string


# Timeout exception
class TimeoutException(Exception):
    pass


# Signal handler for timeout
def timeout_handler(signum, frame):
    raise TimeoutException


manager = Manager()
shared_cache = manager.dict()


def memoized_canonical_form(expression: str, timeout_seconds: int = 3) -> str:
    """
    Compute a canonical form for a mathematical expression using sympy.
    Uses a shared cache across processes for memoization.

    Args:
        expression (str): A LaTeX-formatted mathematical expression.
        timeout_seconds (int): Timeout duration in seconds.

    Returns:
        str: The canonical form of the expression or the original expression as fallback.
    """
    # Check if the result is already cached
    if expression in shared_cache:
        return shared_cache[expression]

    try:
        # Set up the timeout handler
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

        # Parse and simplify the mathematical expression
        parsed_expr = latex2sympy(expression)
        simplified_expr = simplify(parsed_expr)

        # Reset the alarm
        signal.alarm(0)

        canonical_form = latex(simplified_expr)  # Convert back to a string
        shared_cache[expression] = canonical_form  # Cache the result
        return canonical_form
    except TimeoutException:
        # Fallback: Use a stripped version of the input on timeout
        fallback = strip_string(expression)
        shared_cache[expression] = fallback  # Cache the fallback result
        return fallback
    except Exception:
        # Fallback: Use a stripped version of the input on other errors
        fallback = strip_string(expression)
        shared_cache[expression] = fallback  # Cache the fallback result
        return fallback
    finally:
        # Ensure the alarm is turned off
        signal.alarm(0)


def subsample_completions(x: Dict[str, List[Any]], n: int) -> Dict[str, List[Any]]:
    completions = x["completions"]
    agg_scores = x["agg_scores"]
    if len(completions) != len(agg_scores):
        raise ValueError(
            f"The number of completions and agg_scores should be the same. Got {len(completions)} completions and {len(agg_scores)} agg_scores."
        )

    # Take the first n samples, as the completions are ordered in groups of size m e.g [0,0,0,0, 1,1,1,1, 2,2,2,2, ...]
    # We need to ensure these groups are not broken up in order to have a valid comparison at smaller n
    return {
        f"completions@{n}": completions[:n],
        f"agg_scores@{n}": agg_scores[:n],
    }


def extract_completion_answers(
    x: Dict[str, List[Any]], n: int | None = None
) -> Dict[str, List[str]]:
    if n is None:
        return {"preds": [extract_answer(p, "math") for p in x["completions"]]}
    else:
        return {
            f"preds@{n}": [extract_answer(p, "math") for p in x[f"completions@{n}"]]
        }


def compute_naive_pred(x: Dict[str, List[Any]], n: int) -> Dict[str, List[str]]:
    preds = x[f"preds@{n}"]
    scores = x[f"agg_scores@{n}"]
    preds = [
        (p, s) for p, s in sorted(zip(preds, scores), key=lambda x: x[1], reverse=True)
    ]
    return {f"pred_naive@{n}": "\\boxed{" + preds[0][0] + "}"}


def compute_weighted_pred(x: Dict[str, List[Any]], n: int) -> Dict[str, List[str]]:
    preds = x[f"preds@{n}"]
    scores = x[f"agg_scores@{n}"]
    return {
        f"pred_weighted@{n}": "\\boxed{"
        + find_answer_with_largest_sum(preds, scores)
        + "}"
    }


def compute_maj_pred(x: Dict[str, List[Any]], n: int) -> Dict[str, List[str]]:
    preds = x[f"preds@{n}"]
    return {f"pred_maj@{n}": "\\boxed{" + find_majority_answer(preds) + "}"}


def find_answer_with_largest_sum(answers: List[str], scores: List[float]) -> str:
    """
    Groups answers based on their canonical forms and finds the group with the largest sum of scores.

    Args:
        answers (list of str): A list of strings to be grouped.
        scores (list of float): A list of scores corresponding to each string.

    Returns:
        str: The string representing the group with the largest sum of scores.
    """
    if len(answers) == 0 or len(scores) == 0:
        raise ValueError("answers and scores cannot be empty")

    # Grouping using canonical forms
    canonical_groups = defaultdict(
        float
    )  # Stores cumulative scores for each canonical group
    canonical_to_original = {}  # Maps canonical form back to an original answer

    for answer, score in zip(answers, scores):
        # Compute the canonical form
        canonical_form = memoized_canonical_form(answer)

        # Aggregate scores and track the original answer
        canonical_groups[canonical_form] += score
        if canonical_form not in canonical_to_original:
            canonical_to_original[canonical_form] = answer

    # Find the canonical form with the largest cumulative score
    max_canonical = max(canonical_groups, key=canonical_groups.get)
    return canonical_to_original[max_canonical]


def find_majority_answer(answers: List[str]) -> str:
    """
    Groups answers based on their canonical forms and finds the group with the largest number of elements.
    In case of a tie, returns the first occurring group with the largest size.

    Args:
        answers (list of str): A list of strings to be grouped.

    Returns:
        str: The string representing the group with the largest number of elements.

    Example:
        answers = ["a", "b", "a", "c"]
        result = find_majority_answer(answers)
        # result would be "a" since "a" appears most frequently.
    """
    if len(answers) == 0:
        raise ValueError("answers cannot be empty")

    # Group answers using canonical forms
    canonical_groups = defaultdict(int)  # Count occurrences for each canonical form
    canonical_to_original = {}  # Map canonical form back to an original answer

    for answer in answers:
        # Compute the canonical form
        canonical_form = memoized_canonical_form(answer)

        # Increment count for the canonical form
        canonical_groups[canonical_form] += 1

        # Track the original answer for this canonical form
        if canonical_form not in canonical_to_original:
            canonical_to_original[canonical_form] = answer

    # Find the canonical form with the largest count
    max_count = max(canonical_groups.values())
    for canonical_form, count in canonical_groups.items():
        if count == max_count:
            # Return the first occurring group in case of a tie
            return canonical_to_original[canonical_form]


def pass_at_k(n: int, c: int, k: int) -> float:
    """A numerically stable method for calculating an unbiased estimate of pass@k.

    Taken from OpenAI's Codex paper: https://arxiv.org/abs/2107.03374

    Args:
        n (`int`): total number of samples
        c (`int`): number of correct samples
        k (`int`): k in pass@$k$

    Returns:
        `float`: an unbiased estimate of pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def compute_pass_at_k(x, k):
    """
    Computes pass@k for predictions, using canonical forms to group and compare answers.

    Args:
        x (dict): A dictionary containing "preds" (list of predictions) and "answer" (correct answer).
        k (int): The cutoff for pass@k.

    Returns:
        dict: A dictionary containing pass@k results.
    """
    n = len(x["preds"])
    if n == 0:
        raise ValueError("No predictions found")
    if x["answer"] == "":
        raise ValueError("Answer is empty")

    # Compute the canonical form of the correct answer
    canonical_answer = memoized_canonical_form(x["answer"])

    # Compute the count of predictions matching the canonical answer
    c = sum(memoized_canonical_form(pred) == canonical_answer for pred in x["preds"])

    # Calculate pass@k
    return {f"pass@{k}": pass_at_k(n, c, k)}


def compute_level(
    x, metric: Literal["mean_score", "pass@1"], name: str, quintiles: List[float]
) -> Dict[str, int]:
    """Computes the difficulty level (1-5) of a problem based on the given metric and quintiles.

    Easier problems have a a higher metric value, so the levels are reversed (1 is the easiest, 5 is the hardest)."""
    if x[metric] < quintiles[0]:
        return {f"level_{name}": 5}
    elif x[metric] < quintiles[1]:
        return {f"level_{name}": 4}
    elif x[metric] < quintiles[2]:
        return {f"level_{name}": 3}
    elif x[metric] < quintiles[3]:
        return {f"level_{name}": 2}
    else:
        return {f"level_{name}": 1}
