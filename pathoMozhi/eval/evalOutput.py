#%%
import os
import torch
from pathoMozhi import create_model_and_transforms
from torch.cuda.amp import autocast
from helper import load_feats_to_tensor, load_pt_feats_to_tensor

cache_dir=""

device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(device)

model, tokenizer = create_model_and_transforms(
    lang_encoder_path="microsoft/BioGPT-Large",
    tokenizer_path="microsoft/BioGPT-Large",
    cross_attn_every_n_layers=1,
    freeze_lm_embeddings=False,
    use_local_files=False,
    cache_dir=cache_dir,
    cls_type="diagnosis",  # or "None" for no classification
)
tokenizer.padding_side = "left"
model.to(device)
#%%
special_tokens = ["<image>", "<|endofchunk|>"]

for token in special_tokens:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id == tokenizer.unk_token_id:
        print(f"{token} is NOT in the tokenizer vocab.")
    else:
        print(f"{token} is in the tokenizer vocab with ID {token_id}")
#%%
checkpoint_path = "/mnt/bulk-titan/vidhya/pathMozhi/pathoMozhi/pathoMozhi/XATTN48BioGptLargeClsdiagnosisClassifierInit/checkpoint_0.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)

if "model_state_dict" in checkpoint:
    print('Yup we do have a "model_state_dict" key in the checkpoint')
    checkpoint = checkpoint["model_state_dict"]
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}

state = model.load_state_dict(checkpoint, strict=False)
model.eval()

for i, layer in enumerate(model.lang_encoder.gated_cross_attn_layers):
    if layer is None: continue
    print(f'layer {i:02d} attn-gate =', layer.attn_gate.item())
#%%
## Visualisation of Attention Gates
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

gate_values = []
for i, layer in enumerate(model.lang_encoder.gated_cross_attn_layers):
    if layer is not None:
        gate_values.append(round(layer.attn_gate.item(), 4))  # round to 4 decimals
    else:
        gate_values.append(None)  # To keep layer index aligned

df = pd.DataFrame({"Gated Attention": gate_values}, index=[f"Layer {i}" for i in range(len(gate_values))])

plt.figure(figsize=(6, 12))
sns.heatmap(
    df,
    cmap="YlGnBu",
    annot=True,
    fmt=".4f",  # 4 decimal places
    annot_kws={"size": 8},  # optional: smaller font
    linewidths=0.5,
    cbar=True
)
plt.title("Attention Gate Values Across Layers")
plt.xlabel("Model")
plt.ylabel("Transformer Layer")
plt.tight_layout()
plt.show()
#%%
input_folder = "/mnt/bulk-titan/vidhya/pathMozhi/features/conchv1_5/debugStage"
ext = ".h5"  # change to ".pt" if needed

files = sorted([
    os.path.join(input_folder, f) for f in os.listdir(input_folder)
    if f.endswith(ext)
])

def inferReport(feats, prompt: str = "<image>"):
    ids = tokenizer(prompt, return_tensors="pt").to(device)
    tokens = ids["input_ids"][0]
    tokens_decoded = tokenizer.convert_ids_to_tokens(tokens)
    print(f"Tokens: {tokens_decoded}")
    print(f"Contains <image> token?: {model.media_token_id in tokens}")
    print("Vision shape:", feats.shape)
    print("Vision stats → mean:", feats.mean().item(), "std:", feats.std().item())
    with torch.no_grad(), autocast(dtype=torch.float32):
        out = model.generate(
            vision_x=feats.to(device),
            lang_x=ids["input_ids"],
            attention_mask=ids["attention_mask"],
            max_new_tokens=320,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

eval_samples = []
for f in files:
    if ext == ".h5":
        feats = load_feats_to_tensor(f)
    else:
        feats = load_pt_feats_to_tensor(f)
    feats = feats.unsqueeze(1)
    eval_samples.append((f, feats))

import json
from pathlib import Path

results = []
for file_path, feats in eval_samples:
    report = inferReport(feats)
    image_id = Path(file_path).with_suffix(".tiff").name
    results.append({
        "id": image_id,
        "report": report.strip()
    })

# Save to JSON
with open("", "w") as f:
    json.dump(results, f, indent=2)

print("Saved reports to report_outputs.json")
# %%
