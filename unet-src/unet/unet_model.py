""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    """U-Net with an optional layout-classification head.

    `n_layouts=None` (default) keeps the original behaviour exactly: forward()
    returns a single segmentation tensor, so old checkpoints and predict.py
    still work untouched. Pass n_layouts=3 to also predict which layout
    template the page uses (see TEMPLATE_TO_IDX in utils/data_loading.py).
    """

    def __init__(self, n_channels, n_classes, bilinear=False, n_layouts=None, detach_head=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_layouts = n_layouts
        # detach_head=True cuts the gradient path from the classification loss back into
        # the encoder, so segmentation is provably unaffected -- the head then learns from
        # whatever features segmentation already built. Set False to train them jointly
        # (encoder adapts, may score higher, but can move dice either way).
        self.detach_head = detach_head

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        if n_layouts is not None:
            # Layout is a GLOBAL property, so read it off the bottleneck (x5, the widest
            # receptive field). AdaptiveAvgPool collapses any HxW to 1x1, which both keeps
            # the head independent of input size and stops it reading the image dimensions.
            self.layout_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(1024 // factor, n_layouts),
            )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if self.n_layouts is None:
            return logits
        # x5 is the bottleneck; detaching here is what makes the two tasks independent.
        cls_logits = self.layout_head(x5.detach() if self.detach_head else x5)
        return logits, cls_logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)