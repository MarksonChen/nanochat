mkdir -p ~/.cache/nanochat/chatsft_checkpoints/d18
uv run python -c "
from huggingface_hub import hf_hub_download
import shutil, os
repo = 'destinefut/nanochat-d18-sft'
dest = os.path.expanduser('~/.cache/nanochat/chatsft_checkpoints/d18')
tok_dest = os.path.expanduser('~/.cache/nanochat/tokenizer')
os.makedirs(tok_dest, exist_ok=True)
for f in ['model_000483.pt', 'meta_000483.json']:
    shutil.copy(hf_hub_download(repo, f), os.path.join(dest, f))
for f in ['tokenizer.pkl', 'token_bytes.pt']:
    shutil.copy(hf_hub_download(repo, f), os.path.join(tok_dest, f))
print('Done')
"
