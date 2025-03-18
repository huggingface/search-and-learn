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


import dataclasses
import os
import sys
from dataclasses import dataclass
from typing import Any, List, NewType, Optional, Tuple, Union

from transformers import HfArgumentParser

DataClassType = NewType("DataClassType", Any)


class H4ArgumentParser(HfArgumentParser):
    def parse_yaml_and_args(
        self, yaml_arg: str, other_args: Optional[List[str]] = None
    ) -> List[dataclass]:
        """
        Parse a yaml file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (:obj:`str`): the path to the config file used
            other_args (:obj:`List[str]`, `optional`): a list of strings to parse as command line arguments.
                These will look like ['--arg=val', '--arg2=val2'].

        Returns:
            :obj:`List[dataclass]`: a list of dataclasses with the values from the yaml file and the command line
        """
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args_parsed = {}
        for arg in other_args:
            values = arg.split("=")
            if len(values) == 1:
                arg = values[0]  # a boolean value, like --push-to-hub
                value = "True"
            else:
                arg, value = values
            other_args_parsed[arg.strip("-")] = value
        other_args = other_args_parsed
        del other_args_parsed
        used_args = {}

        # overwrite the default/loaded value with the value provided to the command line
        # adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                # add only if in keys
                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type is List[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]

                    # bool of a non-empty string is True, so we manually check for bools
                    if base_type is bool or base_type is Optional[bool]:
                        if val in ["true", "True"]:
                            inputs[arg] = True
                        elif val in ["None", "none"]:
                            inputs[arg] = None
                        else:
                            inputs[arg] = False

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(
                            f"Duplicate argument provided: {arg}, may cause unexpected behavior"
                        )

            obj = data_class(**inputs)
            outputs.append(obj)

        unparsed_args = set(other_args.keys()) - set(used_args.keys())

        if len(unparsed_args) > 0:
            raise ValueError(
                f"The following arguments were not parsed: {unparsed_args}"
            )
        return outputs

    def parse(
        self, allow_extra_keys=False
    ) -> Union[DataClassType, Tuple[DataClassType]]:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a YAML file,
            # let's parse it to get our arguments.
            output = self.parse_yaml_file(
                os.path.abspath(sys.argv[1]), allow_extra_keys=allow_extra_keys
            )
        # parse command line args and yaml file
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
            output = self.parse_yaml_and_args(
                os.path.abspath(sys.argv[1]), sys.argv[2:]
            )
        # parse command line args only
        else:
            output = self.parse_args_into_dataclasses()

        if len(output) == 1:
            output = output[0]
        return output
