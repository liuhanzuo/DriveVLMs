import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor
from drivevlms.models.phi4_bjxx import Phi4MMProcessor, Phi4MMForCausalLM
from drivevlms.build import build_collate_fn

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from functools import partial
from datasets import load_from_disk

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 8192

# Load model.
model_id = 'cutebananas/phi-4-multimodal-finetuned'
model = AutoModelForCausalLM.from_pretrained(
    'cutebananas/phi-4-multimodal-finetuned',
    # 'microsoft/Phi-4-multimodal-instruct',
    torch_dtype=torch.float16, 
    _attn_implementation='sdpa',
    trust_remote_code=True,
)
processor = Phi4MMProcessor.from_pretrained('microsoft/Phi-4-multimodal-instruct')
# processor.chat_template = processor.tokenizer.chat_template

collate_fn = build_collate_fn('drivelm_nus_phi4_collate_fn')
train_collate_fn = partial(collate_fn, processor=processor, dtype = torch.float16)
dataset = load_from_disk('data/DriveLM_nuScenes/split/train')
dataloader = DataLoader(
    dataset,
    batch_size=1,
    collate_fn=train_collate_fn,
    num_workers=0,
    shuffle=False,
)
dataset = dataset.shuffle(seed = 42).select(range(NUM_CALIBRATION_SAMPLES))
dataset = dataset.map(train_collate_fn)

# Recipe
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    sequential_targets=["Phi4MMDecoderLayer"],
    ignore=["lm_head", "re:model.vision_embed_tokens.*"],
)

# Perform oneshot
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
)

SAVE_DIR = "data/models/"+model_id.split("/")[1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)