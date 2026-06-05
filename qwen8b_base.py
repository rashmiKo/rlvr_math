
from vllm import SamplingParams, LLM
from transformers import AutoTokenizer
from datasets import load_dataset
import re

system_prompt = r"""
# Identity

You are an expert assistant who helps users with their math queries.

# Instructions

* Thoroughly explore questions through a systematic thinking process before providing the final precise and accurate solutions.
* Answer the question after thinking step-by-step.
* Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, etc.
* Finally, present the final solution that you deem correct.
* The final solution should be enclosed in a LaTeX box, such as $\\boxed{x}$, where 'x' is the final answer to the problem.

Now, try to solve the following question through the above guidelines:",
# Example 1

Question: Let $x,y$ and $z$ be positive real numbers that satisfy the following system of equations:
\[\log_2\left({x \over yz}\right) = {1 \over 2}\]
\[\log_2\left({y \over xz}\right) = {1 \over 3}\]
\[\log_2\left({z \over xy}\right) = {1 \over 4}\]
Then the value of $\left|\log_2(x^4y^3z^2)\right|$ is $\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$.
Answer: Denote $\log_2(x) = a$, $\log_2(y) = b$, and $\log_2(z) = c$.

Then, we have:
$a-b-c = \frac{1}{2}$,
$-a+b-c = \frac{1}{3}$,
$-a-b+c = \frac{1}{4}$.

Now, we can solve to get $a = \frac{-7}{24}, b = \frac{-9}{24}, c = \frac{-5}{12}$.
Plugging these values in, we obtain $|4a + 3b + 2c| = \frac{25}{8} \implies \boxed{033}$.
$\\boxed{33}$

# Example 2
Question: Let $O(0,0), A(\tfrac{1}{2}, 0),$ and $B(0, \tfrac{\sqrt{3}}{2})$ be points in the coordinate plane. Let $\mathcal{F}$ be the family of segments $\overline{PQ}$ of unit length lying in the first quadrant with $P$ on the $x$-axis and $Q$ on the $y$-axis. There is a unique point $C$ on $\overline{AB}$, distinct from $A$ and $B$, that does not belong to any segment from $\mathcal{F}$ other than $\overline{AB}$. Then $OC^2 = \tfrac{p}{q}$, where $p$ and $q$ are relatively prime positive integers. Find $p + q$.
Answer: Begin by finding the equation of the line $\overline{AB}$: $y = -\sqrt{3}x + \frac{\sqrt{3}}{2}$. Now, consider the general equation of all lines that belong to $\mathcal{F}$. Let $P$ be located at $(a, 0)$ and $Q$ be located at $(0, b)$. With these assumptions, we may arrive at the equation $ay + bx = ab$. However, a critical condition that must be satisfied by our parameters is that $a^2 + b^2 = 1$, since the length of $\overline{PQ} = 1$. We wish to find a point $C$ on $\overline{AB}$ such that $\overline{PQ}$ passes through $C$ if and only if $a = \frac{1}{2}$. Since the property $a^2 + b^2 = 1$ implies that if $a = \frac{1}{2}$, then $\overline{PQ} = \overline{AB}$, we now proceed by finding the intersection of two lines:

a(-\sqrt{3}x + \frac{\sqrt{3}}{2}) + x\sqrt{1-a^2} = a\sqrt{1-a^2}.
After algebraic manipulations, we arrive at the equation:
-a^4 + 2xa^3 + (-4x^2 + 3x + \frac{1}{4})a^2 - 2xa + x^2 = 0.
Note that $a = \frac{1}{2}$ is a solution to this polynomial. Perform polynomial division to eliminate the extraneous solution $a = \frac{1}{2}$. This yields:
-a^3 + (2x - \frac{1}{2})a^2 + (-4x^2 + 4x)a - 2x^2 = 0.
We then plug in $a = \frac{1}{2}$ to find the corresponding values of $x$. This results in the quadratic equation:
16x^2 - 10x + 1 = 0.
This is easily factored to give $x = \frac{1}{8}, \frac{1}{2}$. Since $x = \frac{1}{2}$ corresponds to a point already covered by the horizontal line segment, we discard it. Thus, $x = \frac{1}{8}$ is the only valid solution.
Now, we substitute $x = \frac{1}{8}$ into the equation of line $\overline{AB}$: $y = -\sqrt{3}x + \frac{\sqrt{3}}{2}$, which gives $y = \frac{3\sqrt{3}}{8}$.
The distance from the origin is then given by $\sqrt{\frac{1}{8^2} + \left( \frac{3\sqrt{3}}{8} \right)^2} = \sqrt{\frac{7}{16}}$. Squaring this distance gives $\frac{7}{16}$, so the answer is $\boxed{23}$.
$\\boxed{23}$

""".strip()

def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i+batch_size]

def extract_answer(response):
    if response is None:
        return None

    response = str(response)
    match = re.search(r"\\boxed\{([^}]*)\}", response)
    if match:
        try:
            return int(float(match.group(1)))
        except:
            pass
    try:
        return int(float(response.strip()))
    except:
        return None

def create_prompt(x):
    return (f"""system_prompt
Question: {x['Question']}
Answer: """)

def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Base")
    dataset = load_dataset("gneubig/aime-1983-2024")["train"]
    dataset = list(dataset)

    gen_config = SamplingParams(
            n           = 1,
            max_tokens  = 8192,
            temperature = 0.6,
            top_p       = 0.95,
            skip_special_tokens = False,
    )

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
            model         = "Qwen/Qwen3-8B-Base",
            max_model_len = 8192,
            max_num_seqs  = 4,
    )

    batch_size = 25
    parsed_ques = 0
    printed = 0
    total = 0
    for batch in batchify(dataset, 25):
        prompts = list(map(create_prompt, batch))
        outputs = llm.generate(prompts, gen_config)

        texts = [o.outputs[0].text for o in outputs]
        answers = list(map(extract_answer, [o.outputs[0].text for o in outputs]))
        grnd_truths = list(map(extract_answer, [d["Answer"] for d in batch]))

        #for text, ans, gt in zip(texts, answers, grnd_truths):
        #    print("="*50)
        #    print(f"LLM Solution:\n{text}\n")
        #    print(f"Predicted Answer: {ans}")
        #    print(f"Ground Truth: {gt}")
        #    printed += 1

        count = sum(x == y for x, y in zip(answers, grnd_truths))
        total = total+count
        parsed_ques += len(batch)
        print(f"current accuracy: {total/parsed_ques}")

    print(f"Accuracy qwen8B_base_nochat_template: {total/len(dataset):.3f}")


if __name__ == "__main__":
    main()
## accuracy on using chat_template(mods in fewshot,create_prompt
# and apply_chat_templ): 0.189
