"""
Orca-Math: Grade school math word problems with detailed explanations.
https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k
200K rows, train split only. MIT license.
Used as SFT training data for math reasoning.

The answers are plain text GPT-4-Turbo explanations. We extract the last
number and append "#### <number>" so the model learns the same answer
format as GSM8K (required for eval answer extraction).
"""

import re
from datasets import load_dataset
from tasks.common import Task

LAST_NUM_RE = re.compile(r'(-?\d[\d,]*\.?\d*)')

def extract_last_number(text):
    """Extract the last number from a text string."""
    matches = LAST_NUM_RE.findall(text)
    if matches:
        return matches[-1].replace(',', '')
    return None


class OrcaMath(Task):

    def __init__(self, size=None, **kwargs):
        super().__init__(**kwargs)
        self.ds = load_dataset(
            "microsoft/orca-math-word-problems-200k", split="train"
        ).shuffle(seed=42)
        self.size = size if size is not None else len(self.ds)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return min(self.size, len(self.ds))

    def get_example(self, index):
        row = self.ds[index]
        answer = row['answer']
        # Append #### <number> to match GSM8K answer format
        num = extract_last_number(answer)
        if num is not None:
            answer = answer.rstrip() + f"\n\n#### {num}"
        messages = [
            {"role": "user", "content": row['question']},
            {"role": "assistant", "content": answer},
        ]
        return {"messages": messages}
