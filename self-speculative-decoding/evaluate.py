import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from torch.utils.data import DataLoader
from datasets import load_from_disk
from functools import partial
from drivevlms.build import build_collate_fn
import json
from decoding import infer_input_ids
import numpy as np

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

model.set_skip_layers([0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31],
                      [0, 2, 3, 4, 5, 6, 7, 9, 11, 13, 14, 15, 16, 18, 22, 23, 24, 26, 27, 31])

input = next(iter(dataloader))["input_ids"]

for id, input_id in enumerate(input):
    input_ids = input_id.unsqueeze(0)
    if id == 0:
        th_stop_draft_essg1 = 0.20
        th_stop_draft_essg2 = 0.40
        th_stop_draft_essg3 = 0.60
        th_stop_draft_essg4 = 0.80
        th_stop_draft_essg_autoth  = 0.60
    else:
        th_stop_draft_essg1 = result_essg1['th_stop_draft']
        th_stop_draft_essg2 = result_essg2['th_stop_draft']
        th_stop_draft_essg3 = result_essg3['th_stop_draft']
        th_stop_draft_essg4 = result_essg4['th_stop_draft']
        th_stop_draft_essg_autoth = result_essg_autoth['th_stop_draft']
    print('essg th1: {:.4f}, essg th2: {:.4f}, essg th3: {:.4f}, essg th4: {:.4f}, essg autoth: {:.4f} \n'.format(
    th_stop_draft_essg1, th_stop_draft_essg2, th_stop_draft_essg3, th_stop_draft_essg4, th_stop_draft_essg_autoth))
    result_base = infer_input_ids(model, tokenizer, input_ids, generate_fn='base',
                max_new_tokens=32, do_sample=False, early_stop=True, input_mode=1, use_cache=True)
    print("base")
    result_essg1 = infer_input_ids(model, tokenizer, input_ids, generate_fn='essg', 
                max_new_tokens=32, early_stop=True, max_step_draft=12, 
                th_stop_draft=th_stop_draft_essg1, auto_th_stop_draft=False,
                do_sample=False, do_sample_draft=False, input_mode=1, use_cache=True)
    print("essg1")
    result_essg2 = infer_input_ids(model, tokenizer, input_ids, generate_fn='essg', 
                max_new_tokens=32, early_stop=True, max_step_draft=12, 
                th_stop_draft=th_stop_draft_essg2, auto_th_stop_draft=False,
                do_sample=False, do_sample_draft=False, input_mode=1, use_cache=True)
    print("essg2")
    result_essg3 = infer_input_ids(model, tokenizer, input_ids, generate_fn='essg', 
                max_new_tokens=32, early_stop=True, max_step_draft=12, 
                th_stop_draft=th_stop_draft_essg3,  auto_th_stop_draft=False,
                do_sample=False, do_sample_draft=False, input_mode=1, use_cache=True)
    print("essg3")
    result_essg4 = infer_input_ids(model, tokenizer, input_ids, generate_fn='essg', 
                max_new_tokens=32, early_stop=True, max_step_draft=12, 
                th_stop_draft=th_stop_draft_essg4,  auto_th_stop_draft=False,
                do_sample=False, do_sample_draft=False, input_mode=1, use_cache=True)
    print("essg4")
    result_essg_autoth = infer_input_ids(model, tokenizer, input_ids, generate_fn='essg', 
                max_new_tokens=32, early_stop=True, max_step_draft=12, 
                th_stop_draft=th_stop_draft_essg_autoth, auto_th_stop_draft=True, auto_parameters=[1,0.50,0.90,1e-2,0.90],
                do_sample=False, do_sample_draft=False, input_mode=1, use_cache=True)
    if len(result_base['completion']) < 5 or ('.....' in result_base['completion'][:5]):
        print("too short, skip")
        continue
    results = [
        ('base', result_base),
        ('essg1', result_essg1),
        ('essg2', result_essg2),
        ('essg3', result_essg3),
        ('essg4', result_essg4),
        ('essg_autoth', result_essg_autoth)
    ]
    main_metrics = {'rouge2_base':[], 
                'rouge2_essg1':[], 'rouge2_essg2':[], 'rouge2_essg3':[], 'rouge2_essg4':[], 
                'rouge2_essg_autoth':[],
                'time_base':[], 
                'time_essg1':[], 'time_essg2':[], 'time_essg3':[], 'time_essg4':[], 
                'time_essg_autoth':[],
                'token_time_base':[], 
                'token_time_essg1':[], 'token_time_essg2':[], 'token_time_essg3':[], 'token_time_essg4':[], 
                'token_time_essg_autoth':[],
                'matchness_essg1':[],'num_drafted_tokens_essg1':[],
                'matchness_essg2':[],'num_drafted_tokens_essg2':[],
                'matchness_essg3':[],'num_drafted_tokens_essg3':[],
                'matchness_essg4':[],'num_drafted_tokens_essg4':[],
                'matchness_essg_autoth':[],'num_drafted_tokens_essg_autoth':[]}
    for key, result in results:
        main_metrics['time_' + key].append(result['time'])
        main_metrics['token_time_' + key].append(result['time'] / result['generate_ids'].shape[1])
        if key != 'base':
            main_metrics['matchness_' + key].append(result['matchness'])
            main_metrics['num_drafted_tokens_' + key].append(result['num_drafted_tokens'])
        clip_pred = result['completion'].find("\nArticle:")
        if clip_pred > 0:
            prediction = result['completion'][:clip_pred]
        else:
            prediction = result['completion']
        # rouge_score = rouge.score(prediction, references)
        # rouge_score = 0 # Disabled
        # main_metrics['rouge2_' + key].append(rouge_score['rouge2'].fmeasure)
    metric = {
        # 'mean rouge-2 base':np.mean(main_metrics['rouge2_base']),
        # f'mean rouge-2 essg th {th_stop_draft_essg1}':np.mean(main_metrics['rouge2_essg1']),
        # f'mean rouge-2 essg th {th_stop_draft_essg2}':np.mean(main_metrics['rouge2_essg2']),
        # f'mean rouge-2 essg th {th_stop_draft_essg3}':np.mean(main_metrics['rouge2_essg3']),
        # f'mean rouge-2 essg th {th_stop_draft_essg4}':np.mean(main_metrics['rouge2_essg4']),
        # 'mean rouge-2 essg autoth':np.mean(main_metrics['rouge2_essg_autoth']),
        'mean time base':np.mean(main_metrics['time_base']),
        f'mean time essg th {th_stop_draft_essg1}':np.mean(main_metrics['time_essg1']),
        f'mean time essg th {th_stop_draft_essg2}':np.mean(main_metrics['time_essg2']),
        f'mean time essg th {th_stop_draft_essg3}':np.mean(main_metrics['time_essg3']),
        f'mean time essg th {th_stop_draft_essg4}':np.mean(main_metrics['time_essg4']),
        'mean time essg autoth':np.mean(main_metrics['time_essg_autoth']),
        f'E2E mean speed up essg th {th_stop_draft_essg1}':np.mean(main_metrics['time_base'])/np.mean(main_metrics['time_essg1']),
        f'E2E mean speed up essg th {th_stop_draft_essg2}':np.mean(main_metrics['time_base'])/np.mean(main_metrics['time_essg2']),
        f'E2E mean speed up essg th {th_stop_draft_essg3}':np.mean(main_metrics['time_base'])/np.mean(main_metrics['time_essg3']),
        f'E2E mean speed up essg th {th_stop_draft_essg4}':np.mean(main_metrics['time_base'])/np.mean(main_metrics['time_essg4']),
        'E2E mean speed up essg autoth':np.mean(main_metrics['time_base'])/np.mean(main_metrics['time_essg_autoth']),
        'mean token time base':np.mean(main_metrics['token_time_base']),
        f'mean token time essg th {th_stop_draft_essg1}':np.mean(main_metrics['token_time_essg1']),
        f'mean token time essg th {th_stop_draft_essg2}':np.mean(main_metrics['token_time_essg2']),
        f'mean token time essg th {th_stop_draft_essg3}':np.mean(main_metrics['token_time_essg3']),
        f'mean token time essg th {th_stop_draft_essg4}':np.mean(main_metrics['token_time_essg4']),
        'mean token time essg autoth':np.mean(main_metrics['token_time_essg_autoth']),  
        f'E2E mean token speed up essg th {th_stop_draft_essg1}':np.mean(main_metrics['token_time_base'])/np.mean(main_metrics['token_time_essg1']),
        f'E2E mean token speed up essg th {th_stop_draft_essg2}':np.mean(main_metrics['token_time_base'])/np.mean(main_metrics['token_time_essg2']),
        f'E2E mean token speed up essg th {th_stop_draft_essg3}':np.mean(main_metrics['token_time_base'])/np.mean(main_metrics['token_time_essg3']),
        f'E2E mean token speed up essg th {th_stop_draft_essg4}':np.mean(main_metrics['token_time_base'])/np.mean(main_metrics['token_time_essg4']),
        'E2E mean token speed up essg autoth':np.mean(main_metrics['token_time_base'])/np.mean(main_metrics['token_time_essg_autoth']),          
        f'mean matchness essg th {th_stop_draft_essg1}':np.mean(main_metrics['matchness_essg1']),
        f'mean matchness essg th {th_stop_draft_essg2}':np.mean(main_metrics['matchness_essg2']),
        f'mean matchness essg th {th_stop_draft_essg3}':np.mean(main_metrics['matchness_essg3']),
        f'mean matchness essg th {th_stop_draft_essg4}':np.mean(main_metrics['matchness_essg4']),
        'mean matchness essg autoth':np.mean(main_metrics['matchness_essg_autoth']),
        f'mean num_drafted_tokens essg th {th_stop_draft_essg1}':np.mean(main_metrics['num_drafted_tokens_essg1']),
        f'mean num_drafted_tokens essg th {th_stop_draft_essg2}':np.mean(main_metrics['num_drafted_tokens_essg2']),
        f'mean num_drafted_tokens essg th {th_stop_draft_essg3}':np.mean(main_metrics['num_drafted_tokens_essg3']),
        f'mean num_drafted_tokens essg th {th_stop_draft_essg4}':np.mean(main_metrics['num_drafted_tokens_essg4']),
        'mean num_drafted_tokens essg autoth':np.mean(main_metrics['num_drafted_tokens_essg_autoth']),
    }
    for key, value in metric.items():
        if isinstance(value, float):
            metric[key] = f"{value:.4f}"
    
    print((f'data {id},{metric}'))