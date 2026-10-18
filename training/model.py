import torch
import torch.nn as nn
import torch.nn.functional as F

LEVEL_SIZE = 100
CROP = 32
HALF = 15
NUM_CLASSES = 4
NUM_EXTRA = 7
NUM_OUT = 4


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class NextStatePredictor(nn.Module):
    def __init__(self, hidden_size=256):
        super(NextStatePredictor, self).__init__()

        self.conv = nn.Sequential(
            conv_block(NUM_CLASSES + 2, 32),
            conv_block(32, 64),
            conv_block(64, 128),
        )
        conv_out = 128 * 4 * 4

        self.level_fc = nn.Sequential(nn.Linear(conv_out, hidden_size), nn.ReLU())
        self.extra_fc = nn.Sequential(nn.Linear(NUM_EXTRA, 64), nn.ReLU())

        self.head = nn.Sequential(
            nn.Linear(hidden_size + 64, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, NUM_OUT),
        )

        offsets = (torch.arange(CROP) - HALF) / (CROP - HALF)
        ys, xs = torch.meshgrid(offsets, offsets, indexing="ij")
        self.register_buffer("coords", torch.stack([xs, ys]).unsqueeze(0))

    def forward(self, level, extra):
        x = F.one_hot(level.long(), NUM_CLASSES).permute(0, 3, 1, 2).float()
        x = torch.cat([x, self.coords.expand(x.shape[0], -1, -1, -1)], dim=1)
        x = self.conv(x).flatten(1)
        x = torch.cat([self.level_fc(x), self.extra_fc(extra)], dim=1)
        return self.head(x)

    def loss(self, level, extra, target):
        return F.mse_loss(self(level, extra), target)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = NextStatePredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    B = 16
    level = torch.randint(0, NUM_CLASSES, (B, CROP, CROP))
    extra = torch.randn(B, NUM_EXTRA)
    target = torch.randn(B, NUM_OUT)

    for step in range(5):
        optimizer.zero_grad()
        loss = model.loss(level, extra, target)
        loss.backward()
        optimizer.step()
        print(f"step {step} loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        print(model(level, extra)[:2])
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
