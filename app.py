from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from model import UNetGenerator
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from skimage.color import rgb2lab, lab2rgb
import cv2
import base64

app = FastAPI()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

G = UNetGenerator().to(DEVICE)
state = torch.load("fine_epoch_10.pth", map_location=DEVICE)
if "G" in state:
    state = state["G"]
G.load_state_dict(state)
G.eval()


def preprocess(img_bytes):
    img = Image.open(BytesIO(img_bytes)).convert("RGB").resize((256,256))
    img_rgb = np.array(img)

    lab = rgb2lab(img_rgb).astype("float32")
    L = lab[:,:,0] / 100.0
    L_uint8 = (L * 255).astype("uint8")

    edges = cv2.Canny(L_uint8, 50, 150).astype("float32") / 255.0

    L_edge = torch.tensor(np.stack([L, edges], axis=0)).unsqueeze(0).to(DEVICE)

    return img_rgb, L, edges, L_edge


def colorize(L, edges, L_edge):
    with torch.no_grad():
        ab_pred = G(L_edge).cpu().squeeze(0).numpy().transpose(1,2,0)

    L_lab = (L * 100).astype("float32")
    ab_lab = (ab_pred * 255 - 128).astype("float32")
    lab = np.concatenate([L_lab[...,None], ab_lab], axis=2)
    rgb = (lab2rgb(lab) * 255).astype("uint8")

    return rgb


def encode(img):
    _, buffer = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode("utf-8")


@app.post("/colorize")
async def colorize_api(file: UploadFile = File(...)):
    img_bytes = await file.read()

    img_rgb, L, edges, L_edge = preprocess(img_bytes)
    colorized = colorize(L, edges, L_edge)

    return JSONResponse({
        "original": encode(img_rgb),
        "grayscale": encode((L * 255).astype("uint8")),
        "edges": encode((edges * 255).astype("uint8")),
        "colorized": encode(colorized)
    })
