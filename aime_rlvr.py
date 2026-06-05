import re, json, torch, argparse
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams, LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model


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


