import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from optimum.exporters.onnx import main_export
from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer, QuantizationConfig
from datasets import load_from_disk
from drivevlms.models.phi4_bjxx import Phi4MMProcessor
from drivevlms.build import build_collate_fn
from functools import partial
from torch.utils.data import DataLoader

# -----------------------------
# Constants & Configurations
# -----------------------------
MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
ONNX_EXPORT_DIR = "./onnx_phi4mm"
QUANTIZED_EXPORT_DIR = "./quantized_phi4mm"
NUM_CALIBRATION_SAMPLES = 128

# -----------------------------
# 1. 导出 ONNX 模型
# -----------------------------
print("[1] Exporting ONNX...")
main_export(
    model_name_or_path=MODEL_ID,
    output=ONNX_EXPORT_DIR,
    task="text-generation",
    device="cuda",
    fp16=True,
    trust_remote_code=True
)

# -----------------------------
# 2. 加载校准数据集
# -----------------------------
print("[2] Preparing dataset...")
raw_dataset = load_from_disk('data/DriveLM_nuScenes/split/train')
calib_dataset = raw_dataset.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

processor = Phi4MMProcessor.from_pretrained(MODEL_ID)
collate_fn = build_collate_fn('drivelm_nus_phi4_collate_fn_val')
map_fn = partial(collate_fn, processor=processor, device='cuda')

dataloader = DataLoader(
    calib_dataset,
    batch_size=1,
    collate_fn=map_fn,
    num_workers=0,
    shuffle=False,
)

def collated_generator(dataloader):
    for batch in dataloader:
        yield batch

# -----------------------------
# 3. 量化配置并执行
# -----------------------------
print("[3] Quantizing...")
quantizer = ORTQuantizer.from_pretrained(ONNX_EXPORT_DIR)

quant_config = QuantizationConfig(
    approach="static",
    per_channel=True,
    activation_type="uint8",
    weight_type="int8",
    reduce_range=True
)

quantizer.quantize(
    save_dir=QUANTIZED_EXPORT_DIR,
    calibration_dataset=collated_generator(dataloader),
    quantization_config=quant_config
)

# -----------------------------
# 4. 加载量化模型验证
# -----------------------------
print("[4] Loading quantized model...")
quantized_model = ORTModelForCausalLM.from_pretrained(QUANTIZED_EXPORT_DIR)
print("Quantized model loaded successfully.")
