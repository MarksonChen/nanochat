"""
Orca-Math: Grade school math word problems with detailed explanations.
https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k
200K rows, train split only. MIT license.
Used as SFT training data for math reasoning.
"""

from datasets import load_dataset
from tasks.common import Task


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
        messages = [
            {"role": "user", "content": row['question']},
            {"role": "assistant", "content": row['answer']},
        ]
        return {"messages": messages}
