import torch
from drivevlms_vllm_plugin.phi4.phi4mm import Phi4MMForCausalLM
import argparse
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm
from functools import partial
import argparse
from drivevlms.build import build_collate_fn
import json
from vllm import LLM, SamplingParams, ModelRegistry

@torch.no_grad() 
def main(args):
    print("Loading model...")
    ModelRegistry.register_model("Phi4MMDriveVLMsForCausalLM", Phi4MMForCausalLM)
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
    # model.to(args.device)

    llm = LLM(model='data/models/phi-4-multimodal-finetuned-merged',
        trust_remote_code=True,
        # mm_processor_kwargs={
        #     "padding": "longest",
        # },
        speculative_config={
            # "method": "ngram",
            # "num_speculative_tokens": 5,
            # "prompt_lookup_max": 4,
            "model": "data/models/phi-4-multimodal-finetuned-merged-ssd-pruned",
            "num_speculative_tokens": 4,
        },
        max_model_len=4096,
        gpu_memory_utilization=0.45,
        limit_mm_per_prompt={"image": 8, "audio": 0},
        # disable_log_stats=False,
    )
    print("Loading model done.")
    # generation_config = GenerationConfig.from_pretrained("google/paligemma-3b-pt-224")
    # generation_config = GenerationConfig.from_pretrained('microsoft/Phi-4-multimodal-instruct')

    # prepare dataset
    collate_fn = build_collate_fn(args.collate_fn)
    val_collate_fn = partial(collate_fn, processor=None, device=args.device)
    dataset = load_from_disk(args.data)
    print("Loading dataset done.")
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=val_collate_fn,
        num_workers=0,
        shuffle=False,
    )
    def infer(prompts, image):
        # input_len = inputs["input_ids"].shape[-1]
        # output = model.generate(
        #     **inputs,
        #     max_new_tokens=512,
        #     generation_config=generation_config
        # )
        generation_config = SamplingParams(temperature=1, max_tokens=512, seed=42)
        outputs = llm.generate([
            {
                "prompt": _prompt,
                "multi_modal_data": {"image": _image}
            } for _prompt, _image in zip(prompts, image)],
            sampling_params = generation_config,
        )
        # output = output[:, input_len:]
        # results = processor.batch_decode(output, skip_special_tokens=True)
        return [o.outputs[0].text for o in outputs]

    def flatten(x):
        return x[0] if isinstance(x, list) else x
    
    data_dict = []
    with torch.no_grad():
        cnt = 0
        for batch in tqdm(dataloader):
            cnt += 1
            prompts, image, question, ids = batch
            results = infer(prompts, image)
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