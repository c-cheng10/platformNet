#out: {posx, posy, vx, vy, isDead, isFinished}
#in: {LEVEL (100x100, 4 tile classes: empty, solid, exit, spike), left, right, jump, Px, Py, Vx, Vy}
import torch
import torch.nn as nn
import torch.nn.functional as F

LEVEL_SIZE = 100
NUM_CLASSES = 4
NUM_EXTRA = 7
NUM_OUT = 6


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

        # Level is one-hot encoded to 4 channels, plus 2 coordinate channels (x, y)
        # so the convs know *where* things are, not just what they look like
        self.conv = nn.Sequential(
            conv_block(NUM_CLASSES + 2, 32),  # 100 -> 50
            conv_block(32, 64),               # 50 -> 25
            conv_block(64, 128),              # 25 -> 12
            conv_block(128, 128),             # 12 -> 6
        )
        conv_out = 128 * 6 * 6

        self.level_fc = nn.Sequential(nn.Linear(conv_out, hidden_size), nn.ReLU())
        self.extra_fc = nn.Sequential(nn.Linear(NUM_EXTRA, 64), nn.ReLU())

        self.head = nn.Sequential(
            nn.Linear(hidden_size + 64, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, NUM_OUT),
        )

        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, LEVEL_SIZE), torch.linspace(-1, 1, LEVEL_SIZE), indexing="ij"
        )
        self.register_buffer("coords", torch.stack([xs, ys]).unsqueeze(0))

    def logits(self, level, extra):
        # level: (B, 100, 100) ints in {0, 1, 2, 3}   extra: (B, 7) floats
        x = F.one_hot(level.long(), NUM_CLASSES).permute(0, 3, 1, 2).float()
        x = torch.cat([x, self.coords.expand(x.shape[0], -1, -1, -1)], dim=1)
        x = self.conv(x).flatten(1)
        x = torch.cat([self.level_fc(x), self.extra_fc(extra)], dim=1)
        return self.head(x)

    def forward(self, level, extra):
        out = self.logits(level, extra)
        # first 4 are unbounded (pos/vel), last 2 squashed to 0-1 (isDead, isFinished)
        return torch.cat([out[:, :4], torch.sigmoid(out[:, 4:])], dim=1)

    def loss(self, level, extra, target, mask=None):
        # mask: (B,) 0 for frames whose position/velocity shouldn't be learned (e.g. deaths, which
        # teleport to spawn); they still train isDead / isFinished
        out = self.logits(level, extra)
        regression = F.mse_loss(out[:, :4], target[:, :4], reduction="none").mean(1)
        if mask is None:
            mask = torch.ones_like(regression)
        regression = (regression * mask).sum() / mask.sum().clamp(min=1)
        flags = F.binary_cross_entropy_with_logits(out[:, 4:], target[:, 4:])
        return regression + flags


if __name__ == "__main__":
    # Smoke test on random data
    torch.manual_seed(0)
    model = NextStatePredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    B = 16
    level = torch.randint(0, NUM_CLASSES, (B, LEVEL_SIZE, LEVEL_SIZE))
    extra = torch.randn(B, NUM_EXTRA)
    target = torch.cat([torch.randn(B, 4), torch.randint(0, 2, (B, 2)).float()], dim=1)

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
