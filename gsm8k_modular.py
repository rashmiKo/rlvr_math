
from vllm import SamplingParams, LLM
from transformers import AutoTokenizer
from datasets import load_dataset
import re, argparse

system_prompt = r"""
# Identity
You are a helpful AI assistant. Help the user solve a mathematical query.

# Instruction

*The user will provide you with a math word problem
*You must solve the problem by thinking step by step
*Provide only the final answer at the end of the solution after #### 
*Do not provide any extra text after ### other than the final answer

# Example 1

<user>
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
</user>
<assistant>
Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72
</assistant>

# Example 2
<user>
Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
</user>
<assistant>
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10
</assistant>
""".strip()

def parser_arguments():
	parser = argparse.ArgumentParser(
		description = "This is a modular code for running gsm8k"
	)

	parser.add_argument(
		"--model",
		type = str,
		default = "models/qwen3_8B",
	)
	parser.add_argument(
		"--batch_size",
		type = int,
		default = 25,
	)

	parser.add_argument(
		"--lr",
		type = float,
		default = 1e-6,
		)
	parser.add_argument(
		"--till",
		type = int,
		default = 200
	)

	return parser.parse_args()


def batchify(lst, batch_size):
	for i in range(0, len(lst), batch_size):
		yield lst[i:i+batch_size]

def extract_answer(response):
	match = re.search(r"####\s*([-+]?\d*\.?\d+)", response)

	if match is not None:
		try:
			return int(float(match.group(1)))
		except:
			return None

	return None

def main():

	args = parser_arguments()
	tokenizer = AutoTokenizer.from_pretrained(args.model)
	dataset = load_dataset("openai/gsm8k", "main")["test"]
	dataset = list(dataset)
	dataset = dataset[:args.till]

	gen_config = SamplingParams(
		n           = 1,
		max_tokens  = 8192,
		temperature = 0.6,
		top_p       = 0.95,
		skip_special_tokens = False,
	)

	create_prompt = lambda x: [\
		{"role": "system", "content": system_prompt},\
		{"role": "user", "content": x["question"]},\
	]

	llm = LLM(
		model         = args.model,
		max_model_len = 8192,
		max_num_seqs  = 4,
	)

	total = 0
	batch_size = args.batch_size
	for batch in batchify(dataset, batch_size):
		prompts = list(map(create_prompt, batch))
		outputs = llm.generate(prompts, gen_config)

		texts = [o.outputs[0].text for o in outputs]
		answers = list(map(extract_answer, [o.outputs[0].text for o in outputs]))
		grnd_truths = list(map(extract_answer, [d["answer"] for d in batch]))

		count = sum(x == y for x, y in zip(answers, grnd_truths))
		total = total+count

	print(f"Accuracy of model {args.model}: {total/len(dataset):.3f}") ##accuracy =


if __name__ == "__main__":
	main()
