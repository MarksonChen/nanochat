"""
Sample GSM8K questions and print the model's responses.

Usage:
python -m scripts.gsm8k_sample -i sft -g d18_init -x 10
"""
import argparse
import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K, extract_answer

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--source', type=str, required=True, help="Source of the model: sft|rl")
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-x', '--num-questions', type=int, default=10, help='Number of questions to sample')
parser.add_argument('-t', '--temperature', type=float, default=0.0, help='Temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
parser.add_argument('-m', '--max-new-tokens', type=int, default=512, help='Max new tokens to generate')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'])
args = parser.parse_args()

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
engine = Engine(model, tokenizer)

task = GSM8K(subset="main", split="test")
num_questions = min(args.num_questions, len(task))

num_correct = 0
for i in range(num_questions):
    conversation = task[i]
    question = conversation['messages'][0]['content']

    # Get ground truth
    assistant_parts = conversation['messages'][1]['content']
    gt_text = assistant_parts[-1]['text']
    gt_answer = extract_answer(gt_text)

    # Generate model response
    encoded_prompt = tokenizer.render_for_completion(conversation)
    results, _ = engine.generate_batch(
        encoded_prompt,
        num_samples=1,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    completion = tokenizer.decode(results[0][len(encoded_prompt):])
    pred_answer = extract_answer(completion)
    correct = pred_answer == gt_answer
    num_correct += int(correct)

    mark = "CORRECT" if correct else "WRONG"
    print(f"\n{'='*60}")
    print(f"Question {i+1}/{num_questions}")
    print(f"{'='*60}")
    print(f"Q: {question}")
    print(f"\nModel response:\n{completion}")
    print(f"\nPredicted: {pred_answer} | Ground truth: {gt_answer} | {mark}")

print(f"\n{'='*60}")
print(f"Total: {num_correct}/{num_questions} ({100*num_correct/num_questions:.1f}%)")
