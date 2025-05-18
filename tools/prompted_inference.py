import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig
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
import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ["TORCH_CUDA_ARCH_LIST"] = "sm_90"
@torch.no_grad() 
# prompt_template1 = """
# You are a professional autonomous driving scene understanding model. Carefully analyze the following scene and answer the question.

# Scene description: {scene_description}
# Question: {question}

# Please follow these steps:
# 1. Identify key objects and their relationships in the scene
# 2. Understand the core intention of the question
# 3. Provide a detailed answer based on scene context
# 4. Ensure the answer is accurate, complete, and complies with traffic rules
# """
# prompt_template2 = """
# [System Instruction] You are a multimodal autonomous driving analysis system that needs to integrate visual and textual information.

# [Visual Input] The current scene contains:
# {visual_elements}

# [Question] {question}

# [Response Requirements]
# - First describe key objects and their spatial relationships
# - Then analyze the specific objects involved in the question
# - Finally provide an answer based on scene understanding
# - For prediction questions, explain your reasoning process
# """

# prompt_template3 = """
# [Safety-Critical Instruction] You are handling autonomous driving safety questions. Pay special attention to:

# Question type: {question_type}
# Scene danger level: {danger_level}

# [Question] {question}

# [Response Guidelines]
# 1. First assess potential risks in the scene
# 2. Then analyze the safety implications
# 3. Provide an answer compliant with safety standards
# 4. Suggest preventive measures if necessary
# """

def main(args):
    print("Loading model...")
    # Load model and processor
    # processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
    # model = AutoModelForImageTextToText.from_pretrained("google/paligemma-3b-pt-224")
    model = AutoModelForCausalLM.from_pretrained(
    'cutebananas/phi-4-multimodal-finetuned',
    # 'microsoft/Phi-4-multimodal-instruct',
    torch_dtype=torch.float16, 
    _attn_implementation='sdpa',
    trust_remote_code=True,
    )
    print("Loading Lora")
    model = PeftModel.from_pretrained(model, '/root/autodl-tmp/DriveVLMs/~/lora/model/FULL-2025-05-17_00-22/final_model')
    # model = PeftModel.from_pretrained(model, '/root/lora/small_model/FULL-2025-05-17_09-49/final_model')
    # lora_model_id = "cutebananas/paligemma-finetuned-lora"
    # model = PeftModel.from_pretrained(model, lora_model_id)
    # model = model.merge_and_unload()
    processor = Phi4MMProcessor.from_pretrained('microsoft/Phi-4-multimodal-instruct')
    print("processor loaded")
    # model = AutoModelForCausalLM.from_pretrained(
    # 'cutebananas/phi-4-multimodal-finetuned',
    # # 'microsoft/Phi-4-multimodal-instruct',
    # torch_dtype=torch.float16, 
    # _attn_implementation='sdpa',
    # trust_remote_code=True,
    # )
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
    # collate_fn = build_collate_fn(args.collate_fn)
    # val_collate_fn = partial(collate_fn, processor=processor, device=args.device)
    
    # # Load dataset and select 600 random samples
    # full_dataset = load_from_disk(args.data)
    
    # # Set random seed for reproducibility
    # random_seed = 42
    # torch.manual_seed(random_seed)
    
    # # Get dataset size and check if it has at least 600 samples
    # dataset_size = len(full_dataset)
    # if dataset_size < 600:
    #     print(f"Warning: Dataset only contains {dataset_size} samples, using all available")
    #     num_samples = dataset_size
    # else:
    #     num_samples = 600
    
    # # Create random indices and select subset
    # rand_indices = torch.randperm(dataset_size)[:num_samples].tolist()
    # dataset = full_dataset.select(rand_indices)
    
    # print(f"Selected {len(dataset)} random samples from evaluation set")
    
    # # [Rest of your dataloader and inference code remains the same...]
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=1,
    #     collate_fn=val_collate_fn,
    #     num_workers=0,
    #     shuffle=False,
    # )
    # PROMPT_TEMPLATES = {
    #     "default": "Question: {question}\nBased on the scene, answer:",
    #     "detailed": """
    #     You are a professional autonomous driving scene understanding model. Analyze the scene and answer carefully.

    #     Scene contains these key elements: {scene_elements}
    #     Question: {question}

    #     Please think step by step:
    #     1. Analyze objects and relationships
    #     2. Understand question intent
    #     3. Provide detailed response
    #     """,
    #     "safety": """
    #     [Safety-Critical Question] Pay special attention to traffic safety:
        
    #     Question: {question}
        
    #     Response requirements:
    #     - Assess risk level
    #     - Analyze safety implications
    #     - Provide safety recommendations
    #     """
    # }
    PROMPT_TEMPLATES = {
        "default": "You are a professional autonomous driving scene understanding mode, Q: {question}\nBased on the scene, answer:",  # Simplified template
        "detailed": "Analyze: {question}\nSteps:\n1. Identify objects\n2. Answer:",  # Shorter version
        "safety": "Safety Q: {question}\nConsider risks then answer:"  # Compact safety prompt
    }

    def select_template(question):
        """Select appropriate template based on question content"""
        if not isinstance(question, str):
            question = question[0] if isinstance(question, list) else str(question)
            
        safety_keywords = ["danger", "safety", "collision", "avoid", "risk"]
        question_lower = question.lower()
        
        if any(keyword in question_lower for keyword in safety_keywords):
            return "safety"
        elif len(question.split()) > 8:  # Longer questions get detailed template
            return "detailed"
        return "default"
    
    # def build_prompt(question, scene_elements=""):
    #     """Build the final prompt after template selection"""
    #     if isinstance(question, list):
    #         question = question[0]
            
    #     template_type = select_template(question)
    #     return PROMPT_TEMPLATES[template_type].format(
    #         question=question,
    #         scene_elements=scene_elements
    #     )
    def build_prompt(question, scene_elements=""):
        """Use simplest prompt template always"""
        return PROMPT_TEMPLATES["default"].format(question=question)

    def extract_scene_elements(inputs):
        """
        Extract key scene elements from the model inputs
        Args:
            inputs: Dictionary containing model inputs (pixel_values, input_ids, etc.)
        Returns:
            str: Textual description of key scene elements
        """
        # For nuScenes data, we can extract information from both visual and text inputs
        scene_elements = []
        
        # 1. Extract basic information from input metadata if available
        if 'metadata' in inputs:
            metadata = inputs['metadata']
            if 'location' in metadata:
                scene_elements.append(f"Location: {metadata['location']}")
            if 'weather' in metadata:
                scene_elements.append(f"Weather: {metadata['weather']}")
            if 'time_of_day' in metadata:
                scene_elements.append(f"Time: {metadata['time_of_day']}")
        
        # 2. Extract objects from input text (assuming question/context is in input_ids)
        if 'input_ids' in inputs:
            input_text = processor.decode(inputs['input_ids'][0], skip_special_tokens=True)
            
            # Common nuScenes objects to look for
            nu_scenes_objects = [
                'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic light', 'stop sign', 'lane', 'road', 'intersection'
            ]
            
            detected_objects = [obj for obj in nu_scenes_objects if obj in input_text.lower()]
            if detected_objects:
                scene_elements.append(f"Detected objects: {', '.join(detected_objects)}")
        
        # 3. If we have image features, we could add generic descriptors
        if 'pixel_values' in inputs:
            scene_elements.append("Visual scene contains multiple traffic participants")
        
        return "\n".join(scene_elements) if scene_elements else "General traffic scene"
    def infer(inputs, question):
        """Enhanced inference with prompt engineering"""
        # Ensure question is in proper format
        if isinstance(question, list):
            question = question[0]
            
        scene_elements = extract_scene_elements(inputs)
        enhanced_prompt = build_prompt(question, scene_elements)
        
        # For Phi-4 model, combine prompt with original inputs
        if 'input_ids' in inputs:
            original_text = processor.decode(inputs['input_ids'][0], skip_special_tokens=True)
            combined_input = f"{enhanced_prompt}\n\nContext: {original_text}"
            inputs = processor(text=combined_input, 
                             images=inputs['pixel_values'] if 'pixel_values' in inputs else None,
                             return_tensors="pt").to(args.device)
        
        input_len = inputs["input_ids"].shape[-1]
        output =    model.generate(
            **inputs,
            max_new_tokens=1000,
            generation_config=generation_config,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True
        )
        output = output[:, input_len:]
        return processor.batch_decode(output, skip_special_tokens=True)

    def flatten(x):
        return x[0] if isinstance(x, list) else x
    
    data_dict = []
    with torch.no_grad():
        cnt = 0
        for batch in tqdm(dataloader):
            cnt += 1
            inputs, question, ids = batch
            results = infer(inputs.to(args.device), question)
            data_dict.append(
                {'id': flatten(ids), 'question': flatten(question), 'answer': flatten(results)}
            )
            if not os.path.exists(args.output):
                os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description='DriveLM Inference')
    parser.add_argument("--data", type=str, default="data/DriveLM_nuScenes/split/val")
    parser.add_argument("--collate_fn", type=str, default="drivelm_nus_phi4_collate_fn_val")
    parser.add_argument("--output", type=str, default="data/DriveLM_nuScenes/pe/infer_results_simple.json")
    parser.add_argument("--device", default="cuda", help="Device to run inference")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(parse_args())