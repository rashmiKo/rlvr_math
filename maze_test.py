#this is the maze version of AIME
from vllm import SamplingParams, LLM
from transformers import AutoTokenizer
from datasets import load_dataset
import re

from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators
cfg: MazeDatasetConfig = MazeDatasetConfig(
    name="test", # name is only for you to keep track of things
    grid_n=5, # number of rows/columns in the lattice
    n_mazes=200, # number of mazes to generate
    maze_ctor=LatticeMazeGenerators.gen_dfs, # algorithm to generate the maze
    maze_ctor_kwargs=dict(do_forks=False), # additional parameters to pass to the maze generation algorithm
)

# text format for autoreregressive transformers
from maze_dataset.tokenization import MazeTokenizerModular, TokenizationMode, PromptSequencers
token = m.as_tokens(maze_tokenizer=MazeTokenizerModular(
    prompt_sequencer=PromptSequencers.AOTP(), # many options here
))

system_prompt = r"""
# Identity
You are a helpful AI assistant. Help the user solve the maze.

# Instruction

*The user will provide you with a maze problem
*You must solve the problem by thinking step by step
*Provide only the final answer at the end of the solution after #### 
*Do not provide any extra text after ### other than the final answer

# Example 1

<user>
['<ADJLIST_START>', '(4,3)', '<-->', '(4,4)', ';', '(4,0)', '<-->', '(4,1)', ';', '(2,2)', '<-->', '(3,2)', ';', '(2,0)', '<-->', '(2,1)', ';', '(3,4)', '<-->', '(4,4)', ';', '(3,0)', '<-->', '(2,0)', ';', '(3,3)', '<-->', '(3,2)', ';', '(2,3)', '<-->', '(2,4)', ';', '(4,3)', '<-->', '(4,2)', ';', '(3,4)', '<-->', '(2,4)', ';', '(4,1)', '<-->', '(4,2)', ';', '(3,0)', '<-->', '(4,0)', ';', '(2,1)', '<-->', '(2,2)', ';', '<ADJLIST_END>', '<ORIGIN_START>', '(4,4)', '<ORIGIN_END>', '<TARGET_START>', '(2,4)', '<TARGET_END>']
</user>
<assistant>
['<PATH_START>', '(4,4)', '(3,4)', '(2,4)', '<PATH_END>']
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
    tokenizer = AutoTokenizer.from_pretrained("models/qwen3_8B")
    dataset = load_dataset("openai/gsm8k", "main")["test"]
    dataset = list(dataset)

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
    """
    lambda x: ...

    is equivalent to:
    def create_prompt(x):
        prompt = [ {"role": "system", "content": system_prompt},
                    {"role": "user", "content": x["question"]},
        ]
        return prompt
    """

    """
    map : function, iterable (list, etc.) -> f(x) for x in iterable
    [1, 2, 3] -> [f(1), f(2), f(3)]
    """

    llm = LLM(
            model         = "models/qwen3_8B",
            max_model_len = 8192,
            max_num_seqs  = 4,
    )

    total = 0
    for batch in batchify(dataset, 25):
        prompts = list(map(create_prompt, batch)) ##remove batchify. pass the dataset as an argument
        prompts = [tokenizer.apply_chat_template(p, tokenize = False) for p in prompts]
        outputs = llm.generate(prompts, gen_config)

        answers = list(map(extract_answer, [o.outputs[0].text for o in outputs]))
        grnd_truths = list(map(extract_answer, [d["answer"] for d in batch]))

        count = sum(x == y for x, y in zip(answers, grnd_truths))
        total = total+count

    print(f"Accuracy: {total/len(dataset):.3f}") ##accuracy = 0.922


if __name__ == "__main__":
    main()
