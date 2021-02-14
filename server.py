import argparse
import os.path as osp
import shutil
import tempfile
from flask import Flask, request, jsonify
from test import main
from PIL import Image
import torch
from utils import get_masks, apply_patterns 
app = Flask(__name__)

model = torch.load('checkpoint/unet.pt')


@app.route("/upload", methods=["POST"])
def process_image():
    file1 = request.files['image']
    file2 = request.files['pattern']

    # Read the image via file.stream
    img = Image.open(file1.stream)
    pattern = Image.open(file2.stream)
    model = torch.load('checkpoint/unet.pt')
    masks = get_masks(model, img)
    result = apply_patterns(img, masks, pattern)
    print(len(result))
    return jsonify({'msg': 'success', 'size': [img.width, img.height]})


if __name__ == "__main__":
    app.run(debug=True)