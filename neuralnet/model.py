#out: {dPx, dPy, dVx, dVy} (change from the previous frame, standardized)
#in: {LEVEL crop (32x32 tiles around the player, 4 tile classes: empty, solid, exit, spike),
#     left, right, jump, fracX, fracY, Vx, Vy}
import torch
import torch.nn as nn
import torch.nn.functional as F

LEVEL_SIZE = 100
CROP = 32  # crop window, in tiles
HALF = 15  # the player's tile (under its top-left corner) sits at row/col HALF of the crop
NUM_CLASSES = 4
NUM_EXTRA = 7
NUM_OUT = 4


def conv_block(in_ch, out_ch):
    # conv -> norm -> relu, then halve the resolution
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class NextStatePredictor(nn.Module):
    def __init__(self, hidden_size=256):
        super(NextStatePredictor, self).__init__()

        # The crop is one-hot encoded to 4 channels, plus 2 coordinate channels holding each tile's
        # offset from the player's tile, so the convs know where things are relative to the player
        self.conv = nn.Sequential(
            conv_block(NUM_CLASSES + 2, 32),  # 32 -> 16
            conv_block(32, 64),               # 16 -> 8
            conv_block(64, 128),              # 8 -> 4
        )
        conv_out = 128 * 4 * 4

        self.level_fc = nn.Sequential(nn.Linear(conv_out, hidden_size), nn.ReLU())
        self.extra_fc = nn.Sequential(nn.Linear(NUM_EXTRA, 64), nn.ReLU())

        self.head = nn.Sequential(
            nn.Linear(hidden_size + 64, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, NUM_OUT),
        )

        offsets = (torch.arange(CROP) - HALF) / (CROP - HALF)  # -15/17 ... +16/17
        ys, xs = torch.meshgrid(offsets, offsets, indexing="ij")
        self.register_buffer("coords", torch.stack([xs, ys]).unsqueeze(0))

    def forward(self, level, extra):
        # level: (B, 32, 32) crop, ints in {0, 1, 2, 3}   extra: (B, 7) floats
        x = F.one_hot(level.long(), NUM_CLASSES).permute(0, 3, 1, 2).float()
        x = torch.cat([x, self.coords.expand(x.shape[0], -1, -1, -1)], dim=1)
        x = self.conv(x).flatten(1)
        x = torch.cat([self.level_fc(x), self.extra_fc(extra)], dim=1)
        return self.head(x)

    def loss(self, level, extra, target):
        return F.mse_loss(self(level, extra), target)


if __name__ == "__main__":
    # Smoke test on random data
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
