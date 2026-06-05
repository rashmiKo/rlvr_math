import re, json, torch, argparse
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams, LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model

def parser_arguments():
	parser = argparse.ArgumentParser(
		description = "RLVR training"
	)

	parser.add_argument(
		"--model",
		type = str,
		default = "models/qwen3_8B",
	)
	parser.add_argument(
		"--batch size",
		type = int,
		default = 32,
	)
	parser.add_argument(
		"--lr",
		type = float,
		default = 1e-6,
	)
	return parser.parse_args()

def extract_answer(response):
    match = re.search(r"####\s*([-+]?\d*\.?\d+)", response)

    if match is not None:
        try:
            return int(float(match.group(1)))
        except:
            return None

    return None



system_prompt = r"""
You are a helpful AI assistant.

Solve the math problem step by step.

Rules:
1. Show your reasoning.
2. Put the final answer after ####
3. Nothing should appear after #### except the final answer.

Example:

Question:
15 + 23

Answer:
15 + 23 = 38

#### 38
""".strip()

def build_dataset(tokenizer):
	ds = load_dataset("openai/gsm8k", "main",)["train"]
	def format_example(example): ##here example is one datapoint from the above list
		##This function converts one gsm8k sample into an RLVR training sample
		## example = {"question": "What is 15 + 23?", "answer": "15 + 23 = 38\n#### 38"}
		messages = [
			{
			"role": "system",
			"content": system_prompt
			},
			{
			"role": "user",
			"content": example["question"]
			}
		]
		##This creates the conversations
		prompt = tokenizer.apply_chat_template(
			messages, tokenize = False, add_generation_prompt = True
		)

		return{
			"prompt": prompt,
			"answer": extract_answer(example["answer"])
		}


	ds = ds.map(format_example) #we apply the formatter to the entire dataset
	ds = ds.remove_columns(
		[
			column for column in ds.column_names
			if column not in ["prompt", "answer"]
		]
	)
	return ds

def load_model(args):

	quantization_config = BitsAndBytesConfig(
	load_in_4bit=True,
	bnb_4bit_quant_type="nf4",
	bnb_4bit_compute_dtype=torch.bfloat16,
	bnb_4bit_use_double_quant=True,
	)

	model = AutoModelForCausalLM.from_pretrained(
		args.model,
		quantization_config = quantization_config,
		device_map = "auto",
		torch_dtype=torch.bfloat16,
	)

	peft_config = LoraConfig(
		task_type=TaskType.CAUSAL_LM,
		r=16,                       # Rank dimension
		lora_alpha=32,              # Scaling factor
		lora_dropout=0.05,
		bias="none",
		target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Target all linear layers
	)

	model = get_peft_model(model, peft_config)
	model.print_trainable_parameters()
	return model



def reward_function(completions,answer,**kwargs):
	rewards = []

	for completion, gt in zip(completions,answer):
		prediction = extract_answer(completion)
		if prediction is None:
			rewards.append(0.0)
			continue
		rewards.append(
			1.0 if prediction == gt else 0.0)
		        # --- PRINT EVERYTHING LIVE TO YOUR TERMINAL SCREEN ---
		#print("\n" + "="*50)
		#print(f"🤖 MODEL GENERATION:\n{completion}")
		#print(f"🎯 GROUND TRUTH: {answer}")
		#print("="*50 + "\n")

	return rewards

def main():

	args = parser_arguments()
	tokenizer = AutoTokenizer.from_pretrained(args.model)

	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	model = load_model(args)
	dataset = build_dataset(tokenizer)
	training_dataset = dataset.select(range(200))
	eval_dataset = dataset.select(range(201,300 ))


	grpo_config = GRPOConfig(

		output_dir = "ouputs",
		num_generations = 4,
		max_completion_length = 512,
		per_device_train_batch_size = 1,
		gradient_accumulation_steps = 4, ##????
		bf16 = True,
		logging_steps = 5,

                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},

		per_device_eval_batch_size=4,    # Processes 1 test prompt at a time
		eval_strategy="steps",           # Runs testing on a step interval
		eval_steps=5,                    # Runs evaluation every 5 training step
	)

	import torch
	torch.cuda.empty_cache()

	trainer = GRPOTrainer(
		model = model,
		reward_funcs = reward_function,
		args = grpo_config,
		train_dataset = training_dataset,
		eval_dataset = eval_dataset,

	)

	print("starting training loop...")
	trainer.train()


	print("starting test evaluation...")
	eval_metrics = trainer.evaluate()


	#Print the overall performance output
	print("\n" + "="*40)
	print("FINAL TESTING METRICS (OVERALL PERFORMANCE)")
	# GRPOTrainer maps reward mean to overall dataset generation accuracy
	final_reward = eval_metrics.get("eval_rewards/reward_function/mean", 0.0)
	print(f"   Overall Average Accuracy: {final_reward * 100:.2f}%")




if __name__ == "__main__":
	main()
