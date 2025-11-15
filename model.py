# model.py — final stable Streamlit app for your UNet + ViT model
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

# -------------------------
# Model definitions (from training)
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class SimpleViT(nn.Module):
    def __init__(self, in_channels, img_size, patch_size=16, emb_dim=512, depth=4, n_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        H, W = img_size
        self.patch_size = patch_size
        self.num_patches = (H // patch_size) * (W // patch_size)
        self.patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Linear(self.patch_dim, emb_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, emb_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads,
                                                   dim_feedforward=int(emb_dim * mlp_ratio),
                                                   dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.proj_back = nn.Linear(emb_dim, self.patch_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        # unfold into patches
        x_patches = x.unfold(2, p, p).unfold(3, p, p)           # B, C, H//p, W//p, p, p
        x_patches = x_patches.contiguous().view(B, C, -1, p, p) # B, C, num_patches, p, p
        x_patches = x_patches.permute(0, 2, 1, 3, 4).contiguous()# B, num_patches, C, p, p
        x_patches = x_patches.view(B, self.num_patches, -1)     # B, num_patches, patch_dim
        tokens = self.patch_embed(x_patches) + self.pos_embed   # B, num_patches, emb_dim
        tokens = self.transformer(tokens)
        patches_out = self.proj_back(tokens)                    # B, num_patches, patch_dim
        patches_out = patches_out.view(B, self.num_patches, C, p, p)
        Hp = H // p; Wp = W // p
        patches_out = patches_out.permute(0, 2, 1, 3, 4).contiguous()  # B, C, num_patches, p, p
        patches_out = patches_out.view(B, C, Hp, Wp, p, p)
        patches_out = patches_out.permute(0, 1, 2, 4, 3, 5).contiguous()# B, C, Hp, p, Wp, p
        patches_out = patches_out.view(B, C, H, W)
        return patches_out

class UNetViT(nn.Module):
    def __init__(self, in_channels=3, num_classes=4,
                 filters=(64, 128, 256, 512), bottleneck_channels=512,
                 vit_patch_size=16, vit_emb_dim=512, vit_depth=4, vit_heads=8):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, filters[0])
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck_conv = ConvBlock(filters[3], bottleneck_channels)
        self.vit_patch_size = vit_patch_size
        self.vit_emb_dim = vit_emb_dim
        self.vit_depth = vit_depth
        self.vit_heads = vit_heads
        self.vit = None
        self.up4 = nn.ConvTranspose2d(bottleneck_channels, filters[3], kernel_size=2, stride=2)
        self.dec4 = ConvBlock(filters[3]*2, filters[3])
        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.dec3 = ConvBlock(filters[2]*2, filters[2])
        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(filters[1]*2, filters[1])
        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(filters[0]*2, filters[0])
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(filters[0], num_classes)

    def _init_vit(self, feat_h, feat_w, in_ch):
        # create the ViT module parameters without running any heavy computation
        self.vit = SimpleViT(in_channels=in_ch, img_size=(feat_h, feat_w),
                             patch_size=self.vit_patch_size, emb_dim=self.vit_emb_dim,
                             depth=self.vit_depth, n_heads=self.vit_heads)

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        e4 = self.enc4(p3); p4 = self.pool4(e4)
        b = self.bottleneck_conv(p4)
        B, Cb, Hb, Wb = b.shape
        # initialize vit module lazily if needed
        if self.vit is None:
            if Hb % self.vit_patch_size != 0 or Wb % self.vit_patch_size != 0:
                raise ValueError(f"Bottleneck {(Hb, Wb)} not divisible by patch {self.vit_patch_size}")
            self._init_vit(Hb, Wb, Cb)
            self.vit.to(b.device)
        # combine bottleneck with ViT refined output
        b = b + self.vit(b)
        u4 = self.up4(b); d4 = self.dec4(torch.cat([u4, e4], dim=1))
        u3 = self.up3(d4); d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3); d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2); d1 = self.dec1(torch.cat([u1, e1], dim=1))
        pooled = self.global_avg_pool(d1).squeeze(-1).squeeze(-1)
        output = self.classifier(pooled)
        return output

# -------------------------
# Device and hyperparams (match training)
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training-time hyperparameters you used
TRAIN_FILTERS = (32, 64, 128, 256)
TRAIN_BOTTLENECK = 256
TRAIN_VIT_PATCH = 7
TRAIN_VIT_EMB = 256
TRAIN_VIT_DEPTH = 2
TRAIN_VIT_HEADS = 4
NUM_CLASSES = 4

# -------------------------
# Robust loader (no heavyweight dummy forward)
# -------------------------
@st.cache_resource
def load_model(checkpoint_path=r"C:\Users\affan\Desktop\Brain Tumor\unet_vit_hybrid.pth"):
    # instantiate model with the same hyperparams used during training
    model = UNetViT(
        in_channels=3,
        num_classes=NUM_CLASSES,
        filters=TRAIN_FILTERS,
        bottleneck_channels=TRAIN_BOTTLENECK,
        vit_patch_size=TRAIN_VIT_PATCH,
        vit_emb_dim=TRAIN_VIT_EMB,
        vit_depth=TRAIN_VIT_DEPTH,
        vit_heads=TRAIN_VIT_HEADS
    )

    # Create Vit module parameters (without forward) so state_dict keys exist.
    # We know the expected bottleneck spatial size for input 224:
    # input 224 -> pool1 112 -> pool2 56 -> pool3 28 -> pool4 14 -> bottleneck 14x14
    feat_h = feat_w = 14
    in_ch = TRAIN_BOTTLENECK
    model._init_vit(feat_h, feat_w, in_ch)

    # move model to device
    model = model.to(device)

    # load checkpoint safely
    raw = torch.load(checkpoint_path, map_location=device)
    # if user saved dict with "state_dict" key
    if isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    else:
        state = raw

    # strip possible 'module.' prefix from keys
    new_state = {}
    for k, v in state.items():
        new_k = k
        if k.startswith("module."):
            new_k = k[len("module."):]
        new_state[new_k] = v

    # try strict load first
    try:
        model.load_state_dict(new_state, strict=True)
        st.write("Loaded checkpoint with strict=True")
    except RuntimeError as e:
        # fallback: filter only keys that match shapes
        st.write("Strict load failed:", e)
        filtered = {
            k: v for k, v in new_state.items()
            if k in model.state_dict() and model.state_dict()[k].shape == v.shape
        }
        model.load_state_dict(filtered, strict=False)
        st.write("Loaded filtered checkpoint with strict=False — unmatched keys skipped.")

    model.eval()
    return model

# load model (cached)
model = load_model()

# -------------------------
# Transforms, classes, UI
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize([0.1855, 0.1855, 0.1855], [0.1813, 0.1813, 0.1813])
])

class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

st.title("🧠 Brain Tumor Detection — U-Net + ViT")
st.write("Upload an MRI image to classify the tumor type.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

def predict_image(image: Image.Image):
    img_t = transform(image).unsqueeze(0).to(device)
    # run inference
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1).cpu().squeeze(0)
        pred = int(probs.argmax().item())
        confidences = [float(p) for p in probs]
    return pred, confidences

if uploaded_file is not None:
    try:
        image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Running model..."):
            pred, confidences = predict_image(image)
        st.subheader(f"Prediction: {class_names[pred]}")
        st.write(f"Confidence: {confidences[pred]*100:.2f}%")
        # show bar chart
        st.bar_chart({class_names[i]: confidences[i] for i in range(len(class_names))})
    except Exception as ex:
        st.error(f"Error during prediction: {ex}")
