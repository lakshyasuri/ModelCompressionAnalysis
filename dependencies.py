import torch
from pathlib import Path
from utils import evaluate_model, dynamic_quantization, static_quantization
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.ResNet50 import ResNet50Baseline
import json

supported_engines = torch.backends.quantized.supported_engines
if 'qnnpack' in supported_engines:
    torch.backends.quantized.engine = 'qnnpack'
if 'fbgemm' in supported_engines:
    torch.backends.quantized.engine = 'fbgemm'
saved_model_path = Path("./saved_models/resnet")
saved_model_path.mkdir(parents=True, exist_ok=True)
torch.backends.quantized.engine

train_path = Path('/Users/lakshya/quantization/tiny-imagenet-200/train')
test_path = Path('/Users/lakshya/quantization/tiny-imagenet-200/val/processed_val_2')

transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=train_path, transform=transformations)
test_dataset = datasets.ImageFolder(root=test_path, transform=transformations)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

baseline_model = torch.load(saved_model_path / "resnet50_baseline_model.pth", map_location="cpu",weights_only=False)
baseline_model.eval()

# ===== dynamic quantisation on baseline model. Uses the function inside the utils.py script
quantized_dy_model = dynamic_quantization(baseline_model, (torch.randn(1, 3, 224, 224),))
quantized_dy_model

torch.save(quantized_dy_model.state_dict(), saved_model_path / 'resnet50_base_dy_quant_weights.pth')
torch.save(quantized_dy_model, saved_model_path / 'resnet50_base_dy_quant_model.pth')

quantized_dy_model = torch.load(saved_model_path / "resnet50_base_dy_quant_model.pth", map_location="cpu",weights_only=False)
quantized_dy_model.eval()

baseline_metrics = evaluate_model(baseline_model, test_loader, 'resnet50', batches=20)

baseline_metrics

baseline_dy_metrics = evaluate_model(quantized_dy_model, test_loader, 'resnet50', batches=20)

# ======= Static Quantization. Uses the function inside the utils.py script
calibration_loader = DataLoader(torch.utils.data.Subset(train_dataset, range(1000)), batch_size=64)
base_st_quant_model = static_quantization(baseline_model, (torch.randn(1, 3, 224, 224),),
                                         calibration_loader)
base_st_quant_model

# ======= load the structured + unstructured style pruned ResNet50 model!
sparse_pruned_resnet_model = torch.load(saved_model_path / "resnet50_structured_pruned_SparseML40%_finalized_model.pth",
                                        map_location="cpu",weights_only=False)
sparse_pruned_resnet_model.eval()

all_metrics = {
    "baseline_metrics": baseline_metrics,
    "baseline_dy_metrics": baseline_dy_metrics,
    "baseline_st_metrics": baseline_st_metrics,
    "baseline_sparse_prun_metrics": sparse_prun_base_metrics,
    "sparse_prun_dy_quant_metrics": sparse_prun_dy_quant_metrics,
    "sparse_prun_st_quant_metrics": sparse_prun_st_quant_metrics
}
metrics_folder = Path("./model_metrics/resnet50")
metrics_folder.mkdir(parents=True, exist_ok=True)



for name, metrics in all_metrics.items():
    with (metrics_folder / f"{name}.json").open("w") as file:
        json.dump(metrics, file, indent=1)

