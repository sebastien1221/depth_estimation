import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.models as models
from tqdm import tqdm
from torchvision.models import ResNet18_Weights

torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def bayesian_depth_loss(pred_depth, logvar, target):

    precision = torch.exp(-logvar)

    loss = precision * (pred_depth - target)**2 + logvar

    return torch.mean(loss)

class DepthDataset(Dataset):
    def __init__(self, pt_file):
        data = torch.load(pt_file, weights_only=False)

        images = data["images"]
        depths = data["depths"]

        # Resizing data for faster training time
        images = F.interpolate(images, size=(240,320), mode='bilinear')
        depths = F.interpolate(depths, size=(240,320), mode='bilinear')

        self.images = images
        self.depths = depths

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):

        return {
            "image": self.images[idx],
            "depth": self.depths[idx]
        }


def main():
    dataset = DepthDataset("data/nyu_depth_preprocessed.pt")

    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    batch = next(iter(train_loader))
    print(batch['image'].shape)
    print(batch['depth'].shape)

    plt.figure()
    plt.imshow(batch['image'][0].permute(1, 2, 0).cpu())
    plt.figure()
    plt.imshow(batch['depth'][0].permute(1, 2, 0).cpu())
    plt.show()

    class BayesianDepthModel(nn.Module):

        def __init__(self):
            super().__init__()

            # Encoder (pretrained ResNet)
            resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

            self.encoder_layers = nn.ModuleList([
                nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu),
                nn.Sequential(resnet.maxpool, resnet.layer1),
                resnet.layer2,
                resnet.layer3,
                resnet.layer4
            ])

            # Decoder
            self.up4 = self.up_block(512, 256)
            self.up3 = self.up_block(256, 128)
            self.up2 = self.up_block(128, 64)
            self.up1 = self.up_block(64, 64)
            self.up0 = self.up_block(64, 32)

            # Bayesian output
            self.depth_head = nn.Conv2d(32, 1, 1)
            self.logvar_head = nn.Conv2d(32, 1, 1)

        def up_block(self, in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            f1 = self.encoder_layers[0](x)
            f2 = self.encoder_layers[1](f1)
            f3 = self.encoder_layers[2](f2)
            f4 = self.encoder_layers[3](f3)
            f5 = self.encoder_layers[4](f4)

            x = self.up4(f5)
            x = F.interpolate(x, size=f4.shape[2:], mode='bilinear', align_corners=False)
            x = x + f4

            x = self.up3(x)
            x = F.interpolate(x, size=f3.shape[2:], mode='bilinear', align_corners=False)
            x = x + f3

            x = self.up2(x)
            x = F.interpolate(x, size=f2.shape[2:], mode='bilinear', align_corners=False)
            x = x + f2

            x = self.up1(x)
            x = F.interpolate(x, size=f1.shape[2:], mode='bilinear', align_corners=False)
            x = x + f1

            x = self.up0(x)

            depth = self.depth_head(x)
            logvar = self.logvar_head(x)

            return depth, logvar

    model = BayesianDepthModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 100

    print("Device:", device)
    print("Model device:", next(model.parameters()).device)

    loss_history = []

    for epoch in tqdm(range(num_epochs), position=0, leave=True, desc="Epoch"):

        model.train()
        running_loss = 0
        ## Mixed precision for fasting training
        scaler = torch.amp.GradScaler('cuda')
        for batch in train_loader:

            images = batch["image"].to(device)
            depths = batch["depth"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                pred_depth, logvar = model(images)

                loss = bayesian_depth_loss(pred_depth, logvar, depths)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)

        print(f"\nEpoch {epoch+1}: Loss = {epoch_loss:.4f}")


    ## Evaluation metrics
    model.eval()

    mae_total = 0
    rmse_total = 0
    num_batches = 0

    with torch.no_grad():

        for batch in train_loader:

            images = batch["image"].to(device)
            depths = batch["depth"].to(device)

            pred_depth, logvar = model(images)

            mae = torch.mean(torch.abs(pred_depth - depths))
            rmse = torch.sqrt(torch.mean((pred_depth - depths) ** 2))

            mae_total += mae.item()
            rmse_total += rmse.item()
            num_batches += 1

    print("\nEvaluation Metrics:")
    print(f"MAE  : {mae_total / num_batches:.4f}")
    print(f"RMSE : {rmse_total / num_batches:.4f}")

    ## Training curve

    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


    ## Visualization
    model.eval()

    batch = next(iter(train_loader))

    images = batch["image"].to(device)
    depths = batch["depth"].to(device)

    with torch.no_grad():

        pred_depth, logvar = model(images)

        uncertainty = torch.exp(logvar)

    # Move to CPU
    images = images.cpu()
    depths = depths.cpu()
    pred_depth = pred_depth.cpu()
    uncertainty = uncertainty.cpu()

    num_show = 3

    for i in range(num_show):

        rgb = images[i].permute(1,2,0)
        gt = depths[i,0]
        pred = pred_depth[i,0]
        unc = uncertainty[i,0]
        error = torch.abs(gt - pred)

        plt.figure(figsize=(15,4))

        plt.subplot(1,5,1)
        plt.imshow(rgb)
        plt.title("RGB")
        plt.axis("off")

        plt.subplot(1,5,2)
        plt.imshow(gt, cmap="viridis")
        plt.title("Ground Truth Depth")
        plt.axis("off")

        plt.subplot(1,5,3)
        plt.imshow(pred, cmap="viridis")
        plt.title("Predicted Depth")
        plt.axis("off")

        plt.subplot(1,5,4)
        plt.imshow(error, cmap="inferno")
        plt.title("Absolute Error")
        plt.axis("off")

        plt.subplot(1,5,5)
        plt.imshow(unc, cmap="plasma")
        plt.title("Uncertainty")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()