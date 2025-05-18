import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from torch.utils.data import DataLoader
from datasets import load_from_disk
from functools import partial
from drivevlms.build import build_collate_fn
import json

model = AutoModelForCausalLM.from_pretrained(
    'data/models/phi-4-multimodal-finetuned-merged-ssd-pruned',
    # 'microsoft/Phi-4-multimodal-instruct',
    torch_dtype=torch.float16, 
    _attn_implementation='sdpa',
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained('data/models/phi-4-multimodal-finetuned-merged-ssd-pruned',
                                           padding = "longest",
                                           trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('data/models/phi-4-multimodal-finetuned-merged-ssd-pruned', trust_remote_code=True)

model.to("cuda")

collate_fn = build_collate_fn('drivelm_nus_phi4_collate_fn')
val_collate_fn = partial(collate_fn, processor=processor, device="cuda")
dataset = load_from_disk("data/DriveLM_nuScenes/split/train")
dataloader = DataLoader(
    dataset,
    batch_size=8,
    collate_fn=val_collate_fn,
    num_workers=0,
    shuffle=True
)

torch.manual_seed(42)

input = next(iter(dataloader))

from searching import LayerSkippingSearching

layer_searching = LayerSkippingSearching(model, tokenizer, evaluate_input_ids = input["input_ids"], evaluate_config={"generate_fn": "essg", "max_new_tokens": 32}, input_mode = input["input_mode"])

layer_searching.probe([8,10,15,18,20,24,25,26,27,28,29,30,31,], [])
layer_searching.probe([3, 5, 6, 8, 10, 11, 14, 15, 18, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,], [6, 9, 10, 11, 15, 24, 25, 27, 28,])

attn, mlp = layer_searching.search(600)

json.dump({"attn": attn, "mlp": mlp}, open("ssd.json", "w"))