import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig, AutoTokenizer
from drivevlms.models.phi4_bjxx import Phi4MMProcessor, Phi4MMForCausalLM
from transformers import AutoModelForCausalLM
from peft import PeftModel
import argparse
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm
from functools import partial
import argparse
from drivevlms.build import build_collate_fn
import json
from decoding import infer_input_ids

@torch.no_grad() 
def main(args):
    print("Loading model...")
    # Load model and processor
    # processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
    # model = AutoModelForImageTextToText.from_pretrained("google/paligemma-3b-pt-224")
    # model = PeftModel.from_pretrained(model, '/data2/private-data/zhangn/pretrained/paligemma/FULL-2025-04-29_21-11/final_model')
    # lora_model_id = "cutebananas/paligemma-finetuned-lora"
    # model = PeftModel.from_pretrained(model, lora_model_id)
    # model = model.merge_and_unload()
    # processor = Phi4MMProcessor.from_pretrained('microsoft/Phi-4-multimodal-instruct')
    # model = AutoModelForCausalLM.from_pretrained(
    # 'drivelm-project/phi-4-multimodal-finetuned',
    # # 'microsoft/Phi-4-multimodal-instruct',
    # torch_dtype=torch.float16, 
    # _attn_implementation='sdpa',
    # trust_remote_code=True,
    # )
    # model = Phi4MMForCausalLM.from_pretrained(
        # 'drivelm-project/phi-4-multimodal-finetuned',
        # torch_dtype=torch.float16, 
        # _attn_implementation='sdpa',
        # trust_remote_code=True,
    # )
    # tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-4-multimodal-instruct', trust_remote_code=True)
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
    model.set_skip_layers([0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31],
                      [0, 2, 3, 4, 5, 6, 7, 9, 11, 13, 14, 15, 16, 18, 22, 23, 24, 26, 27, 31])
    model.to(args.device)
    print("Loading model done.")
    # generation_config = GenerationConfig.from_pretrained("google/paligemma-3b-pt-224")
    generation_config = GenerationConfig.from_pretrained('microsoft/Phi-4-multimodal-instruct')

    # prepare dataset
    collate_fn = build_collate_fn(args.collate_fn)
    val_collate_fn = partial(collate_fn, processor=processor, device=args.device)
    dataset = load_from_disk(args.data)
    print("Loading dataset done.")
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=val_collate_fn,
        num_workers=0,
        shuffle=False,
    )
    def infer(inputs):
        input_len = inputs["input_ids"].shape[-1]
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            generation_config=generation_config
        )
        output = output[:, input_len:]
        results = processor.batch_decode(output, skip_special_tokens=True)
        return results

    def flatten(x):
        return x[0] if isinstance(x, list) else x
    
    data_dict = []
    with torch.no_grad():
        cnt = 0
        for batch in tqdm(dataloader):
            cnt += 1
            inputs, question, ids = batch
            # results = infer(inputs.to(args.device))
            results = infer_input_ids(model, tokenizer, inputs["input_ids"].to(args.device), 'base',
                                      max_new_tokens=32, do_sample=False, early_stop=True, input_mode=1, use_cache=True)["completion"]
            data_dict.append(
                {'id': flatten(ids), 'question': flatten(question), 'answer': flatten(results)}
            )

            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description='DriveLM Inference')
    parser.add_argument("--data", type=str, default="data/DriveLM_nuScenes/split/val")
    parser.add_argument("--collate_fn", type=str, default="drivelm_nus_phi4_collate_fn_val")
    parser.add_argument("--output", type=str, default="data/DriveLM_nuScenes/refs/infer_results_21-49.json")
    parser.add_argument("--device", default="cuda", help="Device to run inference")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(parse_args())