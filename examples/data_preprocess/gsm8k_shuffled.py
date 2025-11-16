# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess the GSM8k dataset to parquet format, with shuffled reasoning traces.
"""

import argparse
import os
import random
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    """Extracts the numerical solution from the '####' block."""
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None, f"Could not find solution in: {solution_str}"
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

def extract_trace(answer_str):
    """Extracts the reasoning trace (the part before '####')."""
    return answer_str.split("####")[0].strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/scratch/dkalwar/verl-2llm/data/gsm8k_shuffled")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    random.seed(42)
    
    data_source = "openai/gsm8k"

    # Load the original dataset
    dataset = datasets.load_dataset(data_source, "main")

    # 1. Pre-extract all reasoning traces for both splits
    print("Extracting all reasoning traces from the dataset...")
    train_traces = [extract_trace(d['answer']) for d in dataset["train"]]
    test_traces = [extract_trace(d['answer']) for d in dataset["test"]]
    print(f"Found {len(train_traces)} traces for training and {len(test_traces)} for testing.")

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # 2. Modify the mapping function to accept the list of all traces
    def make_map_fn(split, all_traces):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            # The solution is extracted from the ORIGINAL answer
            solution = extract_solution(answer_raw)

            # 3. Get a random trace from a DIFFERENT datapoint
            num_samples = len(all_traces)
            random_idx = idx
            # Ensure we don't pick the trace from the same example
            while random_idx == idx:
                random_idx = random.randint(0, num_samples - 1)
            
            shuffled_trace = all_traces[random_idx]

            # 4. Reconstruct the new "answer" with the shuffled trace and original solution
            answer_with_shuffled_trace = f"{shuffled_trace}\nThe final answer is \\boxed{{{solution}}}"
            
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "original_index": idx,
                    "shuffled_trace_from_index": random_idx,
                    "answer": answer_with_shuffled_trace, # This is the new content
                    "original_answer": answer_raw, # Keep for reference
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    # 5. Apply the new mapping function to the datasets
    train_dataset = dataset["train"].map(function=make_map_fn("train", train_traces), with_indices=True)
    test_dataset = dataset["test"].map(function=make_map_fn("test", test_traces), with_indices=True)

    # --- Saving the data (same as original script) ---
    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)
    hdfs_dir = args.hdfs_dir

    print(f"Saving shuffled train dataset to {os.path.join(local_dir, 'train.parquet')}")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    
    print(f"Saving shuffled test dataset to {os.path.join(local_dir, 'test.parquet')}")
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
    
    print("\nProcessing complete. Example of a shuffled data point:")
    print(train_dataset[0])