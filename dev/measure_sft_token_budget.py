"""
Measure the token budget for SimpleSpelling + SpellingBee vs Orca-Math,
to determine the Orca-Math subsample size that matches the spelling token budget.

Usage:
    python -m dev.measure_sft_token_budget
"""

from nanochat.tokenizer import get_tokenizer
from tasks.spellingbee import SimpleSpelling, SpellingBee
from tasks.orcamath import OrcaMath

tokenizer = get_tokenizer()

def measure_task(task, name, max_tokens=2048):
    total_tokens = 0
    supervised_tokens = 0
    num_truncated = 0
    n = len(task)
    for i in range(n):
        conv = task[i]
        ids, mask = tokenizer.render_conversation(conv, max_tokens=max_tokens)
        total_tokens += len(ids)
        supervised_tokens += sum(mask)
        # check if the conversation was truncated
        # (render_conversation caps at max_tokens, so if len == max_tokens it was likely truncated)
        if len(ids) == max_tokens:
            num_truncated += 1
    avg_total = total_tokens / n
    avg_supervised = supervised_tokens / n
    print(f"{name}: {n:,} examples")
    print(f"  Total tokens:      {total_tokens:>12,}  (avg {avg_total:.1f}/example)")
    print(f"  Supervised tokens: {supervised_tokens:>12,}  (avg {avg_supervised:.1f}/example)")
    print(f"  Truncated (hit {max_tokens}): {num_truncated:,} ({100*num_truncated/n:.1f}%)")
    return total_tokens, supervised_tokens

print("=" * 70)
print("Measuring SimpleSpelling + SpellingBee token budgets...")
print("=" * 70)

t1, s1 = measure_task(SimpleSpelling(size=200000, split="train"), "SimpleSpelling(200K)")
t2, s2 = measure_task(SpellingBee(size=80000, split="train"), "SpellingBee(80K)")

spelling_total = t1 + t2
spelling_supervised = s1 + s2
print(f"\nCombined spelling:")
print(f"  Total tokens:      {spelling_total:>12,}")
print(f"  Supervised tokens: {spelling_supervised:>12,}")

print("\n" + "=" * 70)
print("Measuring Orca-Math token budget (full 200K)...")
print("=" * 70)

orca = OrcaMath()
orca_total, orca_supervised = measure_task(orca, "OrcaMath(200K)")

# Compute the subsample size to match spelling supervised tokens
avg_orca_supervised = orca_supervised / len(orca)
target_size = int(spelling_supervised / avg_orca_supervised)

print(f"\n{'=' * 70}")
print(f"RESULT")
print(f"{'=' * 70}")
print(f"Spelling supervised tokens: {spelling_supervised:,}")
print(f"Orca-Math avg supervised/example: {avg_orca_supervised:.1f}")
print(f"Recommended OrcaMath(size=N): N = {target_size:,}")
print(f"  (this gives ~{target_size * avg_orca_supervised:,.0f} supervised tokens)")
