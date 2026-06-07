import re, json, torch, argparse
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams, LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
import torch

#this is a file for RLVR Training on the aime dataset

system_prompt = r"""
# Identity

You are an expert assistant who helps users with their math queries.

# Instructions

* Thoroughly explore questions through a systematic thinking process before providing the final precise and accurate solutions.
* Answer the question after thinking step-by-step.
* Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ide>
* Finally, present the final solution that you deem correct.
* The final solution should be enclosed in a LaTeX box, such as $\\boxed{x}$, where 'x' is the final answer to the problem.

Now, try to solve the following question through the above guidelines:",
# Example 1

Question: Let $x,y$ and $z$ be positive real numbers that satisfy the following system of equations:
\[\log_2\left({x \over yz}\right) = {1 \over 2}\]
\[\log_2\left({y \over xz}\right) = {1 \over 3}\]
\[\log_2\left({z \over xy}\right) = {1 \over 4}\]
Then the value of $\left|\log_2(x^4y^3z^2)\right|$ is $\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find >
Answer: Denote $\log_2(x) = a$, $\log_2(y) = b$, and $\log_2(z) = c$.
Then, we have:
$a-b-c = \frac{1}{2}$,
$-a+b-c = \frac{1}{3}$,
$-a-b+c = \frac{1}{4}$.

Now, we can solve to get $a = \frac{-7}{24}, b = \frac{-9}{24}, c = \frac{-5}{12}$.
Plugging these values in, we obtain $|4a + 3b + 2c| = \frac{25}{8} \implies \boxed{033}$.
$\\boxed{33}$


# Example 2
Question: Let $O(0,0), A(\tfrac{1}{2}, 0),$ and $B(0, \tfrac{\sqrt{3}}{2})$ be points in the coordinate plane. Let $\mathcal{F}$ be >
Answer: Begin by finding the equation of the line $\overline{AB}$: $y = -\sqrt{3}x + \frac{\sqrt{3}}{2}$. Now, consider the general >

a(-\sqrt{3}x + \frac{\sqrt{3}}{2}) + x\sqrt{1-a^2} = a\sqrt{1-a^2}.
After algebraic manipulations, we arrive at the equation:
-a^4 + 2xa^3 + (-4x^2 + 3x + \frac{1}{4})a^2 - 2xa + x^2 = 0.
Note that $a = \frac{1}{2}$ is a solution to this polynomial. Perform polynomial division to eliminate the extraneous solution $a = >
-a^3 + (2x - \frac{1}{2})a^2 + (-4x^2 + 4x)a - 2x^2 = 0.
We then plug in $a = \frac{1}{2}$ to find the corresponding values of $x$. This results in the quadratic equation:
16x^2 - 10x + 1 = 0.
This is easily factored to give $x = \frac{1}{8}, \frac{1}{2}$. Since $x = \frac{1}{2}$ corresponds to a point already covered by th>
Now, we substitute $x = \frac{1}{8}$ into the equation of line $\overline{AB}$: $y = -\sqrt{3}x + \frac{\sqrt{3}}{2}$, which gives $>
The distance from the origin is then given by $\sqrt{\frac{1}{8^2} + \left( \frac{3\sqrt{3}}{8} \right)^2} = \sqrt{\frac{7}{16}}$. S>
$\\boxed{23}$
""".strip()

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

def build_dataset(tokenizer):
	ds = load_dataset("gneubig/aime-1983-2024")["train"]
	def format_example(example): ##here example is one datapoint from the above list
		##This function converts one aime sample into an RLVR training sample
		## example = {"question": "What is 15 + 23?", "answer": "15 + 23 = 38\n#### 38"}
		messages = [
			{
			"role": "system",
			"content": system_prompt
			},
			{
			"role": "user",
			"content": example["Question"]
			}
		]
		##This creates the conversations
		prompt = tokenizer.apply_chat_template(
			messages, tokenize = False, add_generation_prompt = True
		)

		return{
			"prompt": prompt,
			"answer": str(extract_answer(example["Answer"]))
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
