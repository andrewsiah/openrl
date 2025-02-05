import os
import yaml
import itertools
from copy import deepcopy
import subprocess
from typing import Dict, Any, List
import json
from datetime import datetime

def load_base_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load the base configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_sweep_configs() -> List[Dict[str, Any]]:
    """Generate different configurations for the sweep."""
    # Define the parameter grid for sweeping
    sweep_params = {
        "model": {
            "name": [
                # "Qwen/Qwen2.5-0.5B-Instruct",
                "Qwen/Qwen2.5-1.5B-Instruct",
                "Qwen/Qwen2.5-3B-Instruct",
                "meta-llama/Llama-3.2-1B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct"
            ]
        },
        "training": {
            "learning_rate": [1e-5, 5e-6],
            "per_device_train_batch_size": [1],
            "gradient_accumulation_steps": [4],
            "num_generations": [16],
        },
        "vllm": {
            "gpu_memory_utilization": [0.7]
        }
    }
    
    # Load base config
    base_config = load_base_config()
    configs = []
    
    # Generate all combinations of parameters
    param_names = []
    param_values = []
    for category, params in sweep_params.items():
        for param_name, values in params.items():
            param_names.append((category, param_name))
            param_values.append(values)
    
    # Generate all combinations
    for values in itertools.product(*param_values):
        config = deepcopy(base_config)
        
        # Create a unique name for this run
        run_name_parts = []
        
        # Update config with new values
        for (category, param_name), value in zip(param_names, values):
            config[category][param_name] = value
            
            # Add to run name if it's an important parameter
            if param_name in ["name", "learning_rate", "num_generations"]:
                if param_name == "name":
                    model_short_name = value.split("/")[-1].split("-")[0]
                    run_name_parts.append(model_short_name)
                else:
                    run_name_parts.append(f"{param_name}_{value}")
        
        # Set unique output directory and run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = "-".join(run_name_parts)
        config["training"]["output_dir"] = f"outputs/{run_name}"
        config["training"]["run_name"] = f"{run_name}-{timestamp}"
        
        configs.append(config)
    
    return configs

def save_sweep_config(config: Dict[str, Any], output_dir: str) -> str:
    """Save a sweep configuration to a file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = os.path.join(output_dir, f"config_{timestamp}.yaml")
    
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    return config_path

def run_training(config_path: str):
    """Run the training script with the given configuration."""
    cmd = [
        "accelerate", "launch",
        "--config_file", "accelerate_config.yaml",
        "train.py",
        "--config", config_path
    ]
    
    subprocess.run(cmd, check=True)

def main():
    """Main function to run the sweep."""
    # Create sweep configs
    configs = generate_sweep_configs()
    
    # Create directory for sweep configs
    sweep_dir = "sweep_configs"
    os.makedirs(sweep_dir, exist_ok=True)
    
    # Save sweep metadata
    sweep_metadata = {
        "timestamp": datetime.now().isoformat(),
        "num_configs": len(configs),
        "configs": configs
    }
    with open(os.path.join(sweep_dir, "sweep_metadata.json"), "w") as f:
        json.dump(sweep_metadata, f, indent=2)
    
    # Run each configuration
    for i, config in enumerate(configs, 1):
        print(f"\nRunning configuration {i}/{len(configs)}")
        print(f"Model: {config['model']['name']}")
        print(f"Learning rate: {config['training']['learning_rate']}")
        print(f"Run name: {config['training']['run_name']}")
        
        # Save config and run training
        config_path = save_sweep_config(config, sweep_dir)
        try:
            run_training(config_path)
        except subprocess.CalledProcessError as e:
            print(f"Error running configuration {i}: {e}")
            continue

if __name__ == "__main__":
    main()
