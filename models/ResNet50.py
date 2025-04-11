import torch
import torchvision.models as models


class ResNet50Baseline(torch.nn.Module):
    def __init__(self, num_classes=200, pretrained_weights_path=None):
        super().__init__()

        # Load pretrained ResNet50
        self.model = models.resnet50(pretrained=False)
        self.model.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.4),
            torch.nn.Linear(self.model.fc.in_features, num_classes)
        )

        # Freeze all layers first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze only the last 3 layers (starting from the end)
        unfreeze_count = 3
        trainable_layers = list(self.model.named_parameters())[-unfreeze_count:]
        for name, param in trainable_layers:
            param.requires_grad = True

        if pretrained_weights_path:
            self.load_state_dict(torch.load(pretrained_weights_path, map_location='cpu'))

    def forward(self, x):
        return self.model(x)
