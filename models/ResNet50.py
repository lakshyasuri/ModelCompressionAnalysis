import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import lightning as L
from sparseml.pytorch.optim import ScheduledModifierManager


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

class StructuredPrunedResNet50FineTune(L.LightningModule):
    def __init__(self, pruned_model, num_classes=200, lr=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['pruned_model'])  # save lr and num_classes only
        self.model = pruned_model

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        acc = self.train_acc(out, y)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        acc = self.val_acc(out, y)
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)


class StructuredThenSparseResNet50(L.LightningModule):
    def __init__(self, pruned_model_path, lr=1e-4, recipe_path=None):
        super().__init__()
        self.save_hyperparameters()

        self.model = torch.load(pruned_model_path, map_location="cpu",weights_only=False)
        self.model.eval()

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=200)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=200)

        self.recipe_path = recipe_path
        self.manager = None

        if recipe_path:
            self.manager = ScheduledModifierManager.from_yaml(recipe_path)
            self.manager.initialize(self.model, optimizer=None)  # Attach optimizer later

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        if self.manager:
            self.manager.initialize(self.model, optimizer)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = self.train_acc(outputs, labels)

        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = self.val_acc(outputs, labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)