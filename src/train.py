# train_grpo.py
import os
import yaml
import json
import argparse
from datetime import datetime
from typing import Dict, Any

import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer


# Parse command line arguments
parser = argparse.ArgumentParser(description="Train GRPO models")
parser.add_argument(
    "--config", type=str, default="config.yaml", help="Path to config file"
)
args = parser.parse_args()

# Load config from specified path
with open(args.config, "r") as f:
    config = yaml.safe_load(f)


class LocalExampleLogger:
    def __init__(self, log_dir: str = "training_logs"):
        # Create logs directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.log_dir, "examples.jsonl")

    def log_example(self, example_dict: Dict[str, Any]):
        """Log a single example to JSONL format"""
        log_dict = {
            "step": example_dict["step"],
            "question": example_dict["question"],
            "true_answer": example_dict["true_answer"],
            "response": example_dict["response"],
            "extracted_response": example_dict["extracted_response"],
            "correct": example_dict["correct"],
            "generation_idx": example_dict["generation_idx"],
        }

        # Append to JSONL
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(log_dict) + "\n")


# Set PyTorch memory allocation configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = config["environment"]["pytorch_cuda_alloc_conf"]

# Set HuggingFace cache directories before any HF imports
cache_dir = config["model"]["cache_dir"]
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")

# Load and prep dataset
SYSTEM_PROMPT = config["prompts"]["system_prompt"]
XML_COT_FORMAT = config["prompts"]["xml_cot_format"]


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]  # type: ignore
    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                # {'role': 'user', 'content': 'What is the largest single-digit prime number?'},
                # {'role': 'assistant', 'content': XML_COT_FORMAT.format(
                #    reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
                #    answer="7"
                # )},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )  # type: ignore
    return data  # type: ignore


dataset = get_gsm8k_questions()
test_dataset = get_gsm8k_questions("test")

# Initialize wandb before training
if config["training"]["report_to"] == "wandb":
    # Remove direct wandb initialization - let Accelerate handle it
    pass


# Reward functions
def count_uncertainty_markers(text: str) -> int:
    markers = [
        "i think",
        "hmm",
        "maybe",
        "perhaps",
        "possibly",
        "wondering",
        "wonder if",
        "not sure",
        "wait",
    ]
    return sum(text.lower().count(marker) for marker in markers)


def count_internal_dialogue_markers(text: str) -> int:
    markers = ["let me think", "let's see", "well...", "come to think of it"]
    return sum(text.lower().count(marker) for marker in markers)


def count_reflective_markers(text: str) -> int:
    markers = [
        "it seems",
        "it appears",
        "i guess",
        "i suppose",
        "i believe",
        "in my view",
    ]
    return sum(text.lower().count(marker) for marker in markers)


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    # Get current step from trainer's state
    current_step = trainer.state.global_step if hasattr(trainer, "state") else 0

    # Initialize logger if not already done
    global example_logger
    if not hasattr(correctness_reward_func, "example_logger"):
        example_logger = LocalExampleLogger()
        correctness_reward_func.example_logger = example_logger

    # Log each example
    for i in range(len(responses)):
        example_dict = {
            "step": current_step,
            "question": q,
            "true_answer": answer[i],
            "response": responses[i],
            "extracted_response": extracted_responses[i],
            "correct": extracted_responses[i] == answer[i],
            "generation_idx": i,  # Which generation attempt this was
        }
        example_logger.log_example(example_dict)

    # Calculate marker counts and correctness for all responses
    is_correct = [r == a for r, a in zip(extracted_responses, answer)]
    uncertainty_counts = [count_uncertainty_markers(r) for r in responses]
    internal_dialogue_counts = [count_internal_dialogue_markers(r) for r in responses]
    reflective_counts = [count_reflective_markers(r) for r in responses]

    # Separate counts for correct and incorrect responses
    correct_indices = [i for i, correct in enumerate(is_correct) if correct]
    incorrect_indices = [i for i, correct in enumerate(is_correct) if not correct]

    # Log metrics using trainer's accelerator
    if hasattr(trainer, "accelerator"):
        metrics = {
            "correctness/correct_count": len(correct_indices),
            "correctness/total_examples": len(responses),
            "correctness/accuracy": len(correct_indices) / len(responses),
            # Total markers across all responses
            "markers/total/uncertainty": sum(uncertainty_counts),
            "markers/total/internal_dialogue": sum(internal_dialogue_counts),
            "markers/total/reflective": sum(reflective_counts),
            # Markers in correct responses
            "markers/correct/uncertainty": sum(
                uncertainty_counts[i] for i in correct_indices
            )
            if correct_indices
            else 0,
            "markers/correct/internal_dialogue": sum(
                internal_dialogue_counts[i] for i in correct_indices
            )
            if correct_indices
            else 0,
            "markers/correct/reflective": sum(
                reflective_counts[i] for i in correct_indices
            )
            if correct_indices
            else 0,
            # Markers in incorrect responses
            "markers/incorrect/uncertainty": sum(
                uncertainty_counts[i] for i in incorrect_indices
            )
            if incorrect_indices
            else 0,
            "markers/incorrect/internal_dialogue": sum(
                internal_dialogue_counts[i] for i in incorrect_indices
            )
            if incorrect_indices
            else 0,
            "markers/incorrect/reflective": sum(
                reflective_counts[i] for i in incorrect_indices
            )
            if incorrect_indices
            else 0,
        }
        trainer.accelerator.log(metrics, step=current_step)

    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


model_name = config["model"]["name"]
output_dir = config["training"]["output_dir"]
run_name = config["training"]["run_name"]

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=config["training"]["learning_rate"],
    adam_beta1=config["training"]["adam_beta1"],
    adam_beta2=config["training"]["adam_beta2"],
    weight_decay=config["training"]["weight_decay"],
    warmup_ratio=config["training"]["warmup_ratio"],
    lr_scheduler_type=config["training"]["lr_scheduler_type"],
    logging_steps=config["training"]["logging_steps"],
    bf16=config["training"]["bf16"],
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
    num_generations=config["training"]["num_generations"],
    max_prompt_length=config["training"]["max_prompt_length"],
    max_completion_length=config["training"]["max_completion_length"],
    num_train_epochs=config["training"]["num_train_epochs"],
    save_steps=config["training"]["save_steps"],
    max_grad_norm=config["training"]["max_grad_norm"],
    report_to=["wandb"]
    if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)
    else [],
    log_on_each_node=False,  # Only log on main node
    use_vllm=config["vllm"]["use_vllm"],
    vllm_gpu_memory_utilization=config["vllm"]["gpu_memory_utilization"],
    vllm_device=config["vllm"]["device"],
    gradient_checkpointing=config["training"]["gradient_checkpointing"],
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=getattr(torch, config["model"]["torch_dtype"]),
    attn_implementation=config["model"]["attn_implementation"],
    device_map=None,
    cache_dir=cache_dir,
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer.pad_token = tokenizer.eos_token

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
    eval_dataset=test_dataset,  # Add test dataset for evaluation
)

# Run training with evaluation
trainer.train()
