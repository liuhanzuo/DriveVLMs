from transformers import AutoModelForCausalLM
import torch
base_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-4-multimodal-instruct", 
    torch_dtype=torch.float16, 
    _attn_implementation='sdpa',
    trust_remote_code=True,)

import safetensors.torch
from tqdm import tqdm

def load_model_from_network_storage(checkpoint_path):
    """
    Load a safetensors model from network storage by first copying to memory
    
    Args:
        checkpoint_path: Path to the safetensors file
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # Read the entire file into memory
    print("Reading file into memory...")
    with open(checkpoint_path, 'rb') as f:
        file_content = f.read()
    print(f"Read {len(file_content) / (1024*1024*1024):.2f}GB into memory")
    
    # Load using safetensors.torch.load
    try:
        print("Loading tensors...")
        tensors = safetensors.torch.load(file_content)
        print(f"Successfully loaded {len(tensors)} tensors")
        return tensors
    except Exception as e:
        print(f"Error loading tensors: {str(e)}")
        raise

fine_tuned_weights = load_model_from_network_storage("data/models/phi-4-multimodal-finetuned/model-00001-of-00002.safetensors")
fine_tuned_weights.update(load_model_from_network_storage("data/models/phi-4-multimodal-finetuned/model-00002-of-00002.safetensors"))

model_state_dict = base_model.state_dict()

# Overwrite weights that are available in the fine-tuned checkpoint
for key in tqdm(fine_tuned_weights):
    if key in model_state_dict:
        model_state_dict[key] = fine_tuned_weights[key]
        tqdm.write(f'Updating {key}\n')

base_model.save_pretrained("data/models/phi-4-multimodal-finetuned-merged")