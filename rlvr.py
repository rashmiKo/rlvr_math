import os, toml, re, sys, json, torch, argparse
from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
from utils.rewards import get_reward_functions

config = toml.load("config.toml")


reward_functions = {
    "math": ["math_acc", "math_format"],
    "maze": ["maze_acc", "maze_fmt"],
}

def parse_arguments():
    parser = argparse.ArgumentParser(
                prog = "accelerate launch --config_file=[accelerate/large_qwen3_8B.yaml] --num_processes 4 --num_machines 1 --rdzv_backend static rlvr.py",
                description = "RLVR LLM on OpenThought and Maze datasets.",
                epilog="BODHI project. ©2026. Soumadeep Saha",
    )
    all_models = [i for i in config["LLM"].keys() if isinstance(config["LLM"][i], dict) and "hf_name" in config["LLM"][i]]

    parser.add_argument(
        '--model',
        choices = all_models,
        type    = str,
        default = "qwen3_8B",
        help    = "The model to use.",
    )
    parser.add_argument(
        '--mode',
        choices = ["MATH", "MAZE"],
        type    = str,
        default = "MATH",
    )
    parser.add_argument(
        '--vllm_host',
        type = str,
        default = None,
        help = "vLLM url."
    )
    args = parser.parse_args(sys.argv[1:])
    return args

def main():
    args        = parse_arguments()
    model_path  = config["LLM"][args.model]["base_model"]    # Distilled model
    save_path   = f"{config['LLM'][args.model]['save_dir']}_{args.mode.lower()}"
    dataset     = load_from_disk(config["datasets"]["dataset"])
    dataset     = dataset.filter(lambda example: example["ability"] == args.mode)

    is_colocate = args.vllm_host is None
    vllm_mode   = "colocate" if is_colocate else "server"

    training_args = GRPOConfig(
        output_dir                  = save_path,
        num_generations             = config["train"]["group_size"],
        max_completion_length       = 4096,
        per_device_train_batch_size = config["train"]["per_device_batch"],
        gradient_accumulation_steps = config["train"]["accumulate"],
        loss_type                   = "dr_grpo",
        epsilon                     = 0.2,
        epsilon_high                = 0.28,
        temperature                 = 1.0,
        use_vllm                    = True,
        vllm_mode                   = vllm_mode,
        vllm_server_host            = args.vllm_host.strip() if not is_colocate else None,
        vllm_gpu_memory_utilization = 0.4 if is_colocate else None,
        learning_rate               = config["train"]["lr"],
        num_train_epochs            = config["train"]["epochs"],
        save_strategy               = "steps",
        save_steps                  = 100,
        bf16                        = True,
        logging_steps               = 5,
    )
    
    trainer = GRPOTrainer(
        model            = model_path,
        reward_funcs     = get_reward_functions(reward_functions[args.mode.lower()]),
        args             = training_args,
        train_dataset    = dataset,
    )

    if "gemma" in model_path:
        original_forward = trainer.model.forward
        def gemma_forward_wrapper(*args, **kwargs):
            if "token_type_ids" not in kwargs and "input_ids" in kwargs:
                kwargs["token_type_ids"] = torch.zeros_like(kwargs["input_ids"])
            return original_forward(*args, **kwargs)
        trainer.model.forward = gemma_forward_wrapper

    trainer.train()#resume_from_checkpoint = True)

if __name__ == "__main__":
    main()
