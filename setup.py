# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

extras = {}
extras["quality"] = ["ruff", "isort"]
extras["tests"] = ["pytest"]
extras["dev"] = ["vllm==0.6.3"] + extras["quality"] + extras["tests"]
extras["trl"] = "trl @ git+https://github.com/huggingface/trl.git"

install_requires = [
    "accelerate",
    "pebble",  # for parallel processing
    "latex2sympy2==1.9.1",  # for MATH answer parsing
    "word2number",  # for MATH answer parsing
    "transformers>=4.47.0",
    "fastapi",
    "hf_transfer",
]

setup(
    name="search-and-learn",
    version="0.1.0",
    author="The Hugging Face team (past and future)",
    author_email="lewis@huggingface.co",
    description="A tool for search-based methods on llms",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/huggingface/search-and-learn",
    keywords="nlp deep learning mcts",
    license="Apache",
    package_dir={"": "src"},
    packages=find_packages("src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10.9",
    install_requires=install_requires,
    extras_require=extras,
    include_package_data=True,
)
