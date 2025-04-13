import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import torch.quantization as tq
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, QConfigMapping
from torch.ao.quantization.observer import PlaceholderObserver
from torch.ao.quantization import PerChannelMinMaxObserver, HistogramObserver
import copy
import torch.nn.utils.prune as prune
import torch_pruning as tp
import torch.nn as nn
from torchvision import datasets, transforms
def evaluate_model(model: torch.nn.Module, data_loader: DataLoader, model_name: str,
                   batches: int = None, high_granularity=False):
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    # warm-up
    dummy_input = torch.randn(1, 1, 28, 28).to(device) if model_name == 'lenet5' \
        else torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        _ = model(dummy_input)

    all_losses, all_top1_acc, inference_times = [], [], []
    true_labels, predicted_labels = [], []
    top1_correct = 0
    top5_correct = 0
    total_samples = 0
    total_loss = 0
    i = 0
    batch_size = data_loader.batch_size
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad(), tqdm(total=len(data_loader)) as pbar:
        start_time = time.time()
        for images, labels in data_loader:
            if i == batches:
                break

            images = images.to(device)
            labels = labels.to(device)

            batch_start = time.perf_counter()
            outputs: torch.tensor = model(images)
            batch_end_time = time.perf_counter()

            total_samples += labels.size(0)
            loss = criterion(outputs, labels)
            _, top1_preds = outputs.topk(1, 1, True, True)
            _, top5_preds = outputs.topk(5, 1, True, True)

            current_top1 = top1_preds.eq(labels.view(-1, 1)).sum().item()
            current_top5 = top5_preds.eq(labels.view(-1, 1)).sum().item()
            top1_correct += current_top1
            top5_correct += current_top5
            total_loss += loss.item()

            if high_granularity:
                all_top1_acc.append(current_top1 / labels.size(0))
                all_losses.append(loss.item())
                inference_times.extend(
                    [(batch_end_time - batch_start) * 1000 / images.size(0)] * images.size(0)
                )
                true_labels.extend(labels.tolist())
                predicted_labels.extend(top1_preds.squeeze().tolist())

            pbar.update(1)
            pbar.set_postfix({
                'Loss': f"{total_loss / total_samples:.4f}",
                'Top1': f"{100 * top1_correct / total_samples:.2f}%"
            })
            i += 1
    end_time = time.time()

    # check last batch's size because most probably, it's not an equal split
    if labels.size(0) != batch_size:
        new_last_loss = loss.item() * labels.size(0) / batch_size
        total_loss -= loss.item()
        total_loss += new_last_loss
    total_inference_time = end_time - start_time

    return {
        "top1_acc": top1_correct / total_samples,
        "top5_acc": top5_correct / total_samples,
        "total_inference_time": total_inference_time,
        "average_inference_time": total_inference_time / i,
        "average_loss": total_loss / i,
        "total_batches": i,
        "all_losses": all_losses,
        "all_top1_acc": all_top1_acc,
        "inference_times": inference_times,
        "true_labels": true_labels,
        "predicted_labels": predicted_labels
    }


def quant_model_prep(example_input: tuple[torch.Tensor], model: torch.nn.Module, qconfig_mapping: QConfigMapping):
    return prepare_fx(
        copy.deepcopy(model),
        qconfig_mapping,
        example_input
    )


def dynamic_quantization(model: torch.nn.Module, example_input: tuple[torch.Tensor]):
    if not model.training:
        model.eval()
    dynamic_qconfig = tq.QConfig(
        activation=PlaceholderObserver.with_args(
            dtype=torch.quint8,
            quant_min=0,
            quant_max=255,
            is_dynamic=True
        ),
        weight=PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric
        )
    )
    qconfig_mapping = QConfigMapping()
    qconfig_mapping.set_global(dynamic_qconfig)
    quantized_model = convert_fx(quant_model_prep(example_input, model, qconfig_mapping))
    return quantized_model


def stat_quant_calibration(model: torch.fx.GraphModule, data_loader: DataLoader):
    if not model.training:
        model.eval()
    with torch.no_grad(), tqdm(total=len(data_loader)) as pbar:
        for image, target in data_loader:
            model(image)
            pbar.update(1)
    return model


def static_quantization(model: torch.nn.Module, example_input: tuple[torch.Tensor], calibration_data: DataLoader):
    custom_config = tq.QConfig(
        activation=HistogramObserver.with_args(
            reduce_range=False
        ),
        weight=PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric
        )
    )
    qconfig_mapping = QConfigMapping() \
        .set_global(custom_config) \
        .set_object_type(torch.nn.Conv2d, custom_config) \
        .set_object_type(torch.nn.Linear, custom_config) \
        .set_object_type(torch.nn.ReLU, custom_config)

    stat_quant_prep_mod = quant_model_prep(example_input, model, qconfig_mapping)
    stat_quant_prep_mod = stat_quant_calibration(stat_quant_prep_mod, calibration_data)
    return convert_fx(stat_quant_prep_mod)

def unstructured_prune(model, sparsity=0.3, layer_scope="both"):
    """
    Applies unstructured magnitude-based pruning to Conv2d and/or Linear layers.
    """
    device = torch.device("cpu")
    model = model.to(device)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and layer_scope in ["conv", "both"]:
            prune.l1_unstructured(module, name='weight', amount=sparsity)
            prune.remove(module, 'weight')
        elif isinstance(module, nn.Linear) and layer_scope in ["fc", "both"]:
            prune.l1_unstructured(module, name='weight', amount=sparsity)
            prune.remove(module, 'weight')
    return model

# for structured taylor 
def get_train_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
def structured_prune(
    model,
    method="magnitude",
    sparsity=0.3,
    layer_scope="both",
    example_input=torch.randn(1, 1, 28, 28)
):
    """
    Applies structured pruning using Torch-Pruning library.
    """
    device = torch.device("cpu")
    model = model.to(device)

    # === Set importance
    if method == "magnitude":
        importance = tp.importance.GroupMagnitudeImportance(p=2)
    elif method == "taylor":
        train_loader = get_train_loader()
        importance = tp.importance.GroupTaylorImportance()
        model.train()
        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)
        loss = nn.CrossEntropyLoss()(model(images), labels)
        loss.backward()
        model.eval()
    else:
        raise ValueError("Unsupported method. Use 'magnitude' or 'taylor'.")

    # === Layer filtering
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == 10:
            ignored_layers.append(m)
        if layer_scope == "conv" and isinstance(m, nn.Linear):
            ignored_layers.append(m)
        if layer_scope == "fc" and isinstance(m, nn.Conv2d):
            ignored_layers.append(m)

    # === Prune
    pruner = tp.pruner.BasePruner(
        model,
        example_input,
        importance=importance,
        pruning_ratio=sparsity,
        ignored_layers=ignored_layers,
        round_to=1
    )
    tp.utils.print_tool.before_pruning(model)
    pruner.step()
    tp.utils.print_tool.after_pruning(model)

    return model