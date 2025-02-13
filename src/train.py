# train_grpo.py
import os
import yaml
import json
import argparse
from datetime import datetime
from typing import Dict, Any
import logging

import tqdm
import numpy as np
import re
import torch
import torch.distributed as dist
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
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
    def __init__(self):
        """Initialize the logger with paths for training log and evaluation results."""
        # Only create logs on the main process
        self.is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        
        # Initialize all attributes as None first
        self.logger = None
        self.log_dir = None
        self.log_path = None
        self.eval_path = None
        self.examples_path = None
        
        if self.is_main_process:
            # Create training logs directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = os.path.join("training_logs", timestamp)
            os.makedirs(self.log_dir, exist_ok=True)

            # Set up file paths
            self.log_path = os.path.join(self.log_dir, "training.log")
            self.eval_path = os.path.join(self.log_dir, "eval_results.jsonl")
            self.examples_path = os.path.join(self.log_dir, "examples.jsonl")

            # Set up logging
            self.logger = logging.getLogger("training")
            self.logger.setLevel(logging.INFO)
            
            # File handler
            file_handler = logging.FileHandler(self.log_path)
            file_handler.setLevel(logging.INFO)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # Add handlers
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def log_example(self, example_dict: Dict[str, Any]):
        """Log a single example to the examples file."""
        if self.is_main_process:
            with open(self.examples_path, "a") as f:
                f.write(json.dumps(example_dict) + "\n")


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
    data = load_dataset("openai/gsm8k", "main", split=split)  # type: ignore
    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [
                {"role": "user", "content": SYSTEM_PROMPT},
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
    
    # Set the format for PyTorch
    data = data.with_format("torch")
    return data  # type: ignore


# Load datasets
dataset = get_gsm8k_questions("train")
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
            # Average lengths
            "length/correct_avg": sum(len(responses[i].split()) for i in correct_indices) / len(correct_indices) if correct_indices else 0,
            "length/incorrect_avg": sum(len(responses[i].split()) for i in incorrect_indices) / len(incorrect_indices) if incorrect_indices else 0,
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
        for key, value in metrics.items():
            trainer._metrics[key].append(value)

    return [config["rewards"]["correctness"]["correct"] if r == a else config["rewards"]["correctness"]["incorrect"] 
            for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [config["rewards"]["integer_check"]["is_digit"] if r.isdigit() else config["rewards"]["integer_check"]["not_digit"] 
            for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\n<answer>.*?</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [config["rewards"]["strict_format"]["valid"] if match else config["rewards"]["strict_format"]["invalid"] 
            for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [config["rewards"]["soft_format"]["valid"] if match else config["rewards"]["soft_format"]["invalid"] 
            for match in matches]


def count_xml(text) -> float:
    count = 0.0
    per_tag_reward = config["rewards"]["xml_count"]["per_tag"]
    trailing_penalty = config["rewards"]["xml_count"]["trailing_penalty"]
    
    if text.count("<think>\n") == 1:
        count += per_tag_reward
    if text.count("\n</think>\n") == 1:
        count += per_tag_reward
    if text.count("\n<answer>\n") == 1:
        count += per_tag_reward
        count -= len(text.split("\n</answer>\n")[-1]) * trailing_penalty
    if text.count("\n</answer>") == 1:
        count += per_tag_reward
        count -= (len(text.split("\n</answer>")[-1]) - 1) * trailing_penalty
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


model_name = config["model"]["name"]
output_dir = config["training"]["output_dir"]
run_name = config["training"]["run_name"]

class CustomGRPOTrainer(GRPOTrainer):
    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"
    ):
        # Get logger and example_logger for training logs
        logger = correctness_reward_func.example_logger.logger if hasattr(correctness_reward_func, "example_logger") else None
        example_logger = correctness_reward_func.example_logger if hasattr(correctness_reward_func, "example_logger") else None
        
        if logger is not None:
            logger.info("Starting evaluation...")
        
        # Set seed for evaluation
        if hasattr(self.args, "seed") and self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.args.seed)
        
        tokenized_samples = tokenize_validation(
            self.processing_class, 
            self.eval_dataset, 
            self.args.max_prompt_length
        )
        eval_acc, prompts_and_responses = generate_gsm8k(
            self.model, 
            self.processing_class, 
            tokenized_samples, 
            self.args.per_device_eval_batch_size, 
            self.args.max_completion_length
        )

        output = {
            f"{metric_key_prefix}_accuracy": eval_acc,
            "epoch": self.state.epoch,
            "step": self.state.global_step,
            "prompts_and_responses": prompts_and_responses,  # Include the prompts and responses
            "timestamp": datetime.now().isoformat()
        }
        
        # Write eval results directly to JSONL if on main process
        if hasattr(correctness_reward_func, "example_logger") and correctness_reward_func.example_logger.is_main_process:
            eval_path = correctness_reward_func.example_logger.eval_path
            with open(eval_path, "a") as f:
                f.write(json.dumps(output) + "\n")

        # Remove prompts_and_responses from metrics logged to wandb
        wandb_output = {k: v for k, v in output.items() if k != "prompts_and_responses"}
        self.log(wandb_output)
        
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, wandb_output
        )

        return output

def tokenize_validation(tokenizer, samples, max_prompt_length):
    tokenized_samples = []
    for sample in samples:
        prompt = sample["prompt"]
        answer = sample['answer']
        ids = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            truncation=True,
            max_length=max_prompt_length,
        )
        tokenized_samples.append((ids, answer))
    return tokenized_samples

def generate_gsm8k(
    model,
    tokenizer,
    tokenized_samples,
    batch_size,
    max_completion_length
):
    device = model.device
    predictions = []
    prompts_and_responses = []
    
    # Set evaluation to deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Get world size and rank for distributed processing
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Split samples across GPUs
    samples_per_gpu = len(tokenized_samples) // world_size
    start_idx = rank * samples_per_gpu
    end_idx = start_idx + samples_per_gpu if rank != world_size - 1 else len(tokenized_samples)
    local_samples = tokenized_samples[start_idx:end_idx]
    
    generation_config = GenerationConfig(
        max_new_tokens=max_completion_length,
        do_sample=False,
        num_beams=1,  # Ensure deterministic behavior with greedy decoding
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,  # Enable KV-cache for faster generation
    )
    
    model.eval()
    count = len(local_samples)
    
    # Process in fixed size batches
    status = tqdm.tqdm(range(0, count, batch_size), desc=f"Rank {rank} Evaluating", disable=rank != 0)
    for i in status:
        # Ensure we don't exceed the dataset size
        end_idx = min(i + batch_size, count)
        batches = local_samples[i:end_idx]
        with torch.inference_mode():
            longest = max(len(b[0]) for b in batches)

            # pad to longest on left side for decoder
            padded_input_ids = torch.stack([
                torch.tensor([tokenizer.pad_token_id] * (longest - len(ids)) + ids)
                for ids, _ in batches
            ]).to(device)
            # ignore pad token when generating
            attn_mask = torch.stack([
                tokens.ne(tokenizer.pad_token_id) for tokens in padded_input_ids
            ]).to(device)

            output = model.generate(
                input_ids=padded_input_ids,
                attention_mask=attn_mask,
                generation_config=generation_config,
            )

            for j, generated in enumerate(output):
                # Get the prompt
                prompt = tokenizer.decode(padded_input_ids[j], skip_special_tokens=True)
                # Get the response
                response = tokenizer.decode(
                    generated[len(padded_input_ids[j]):], skip_special_tokens=True
                )

                prediction = extract_xml_answer(response)
                predictions.append(batches[j][1] == prediction)
                
                # Store prompt and response
                prompts_and_responses.append({
                    "prompt": prompt,
                    "response": response,
                    "prediction": prediction,
                    "true_answer": batches[j][1],
                    "is_correct": batches[j][1] == prediction,
                    "rank": rank  # Add rank information
                })

                if rank == 0:  # Update progress only on main process
                    status.set_description(f"Local Correct: {sum(predictions)}/{len(predictions)}")
    
    # Gather results from all processes
    if dist.is_initialized():
        # Gather predictions
        gathered_predictions = [None] * world_size
        dist.all_gather_object(gathered_predictions, predictions)
        
        # Gather prompts and responses
        gathered_prompts_responses = [None] * world_size
        dist.all_gather_object(gathered_prompts_responses, prompts_and_responses)
        
        # Combine results on all processes
        all_predictions = []
        all_prompts_responses = []
        for pred_list in gathered_predictions:
            all_predictions.extend(pred_list)
        for pr_list in gathered_prompts_responses:
            all_prompts_responses.extend(pr_list)
            
        return np.mean(all_predictions), all_prompts_responses
    
    return np.mean(predictions), prompts_and_responses

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
    # Add evaluation settings
    do_eval=True,
    eval_steps=config["training"].get("eval_steps", 50),
    per_device_eval_batch_size=config["training"].get("per_device_eval_batch_size", 32),
    eval_strategy="steps",
    # Add deterministic settings - but only if not using VLLM
    seed=42,  # Fixed seed for reproducibility
    data_seed=42,  # Fixed seed for data sampling
    full_determinism=not config["vllm"]["use_vllm"],  # Only enable full determinism when not using VLLM
    beta=config["training"]["beta"] # KL coefficient
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

trainer = CustomGRPOTrainer(
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

# After creating the trainer but before train()
initial_metrics = trainer.evaluate()
print("Initial evaluation metrics:", initial_metrics)

if hasattr(correctness_reward_func, "example_logger"):
    logger = correctness_reward_func.example_logger.logger
    logger.info("Starting training...")

# Then start training as normal
trainer.evaluate()
trainer.train()

if hasattr(correctness_reward_func, "example_logger"):
    logger = correctness_reward_func.example_logger.logger
    logger.info("Training completed!")
