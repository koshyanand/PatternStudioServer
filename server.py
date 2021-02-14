from flask import Flask, request, jsonify
from PIL import Image
import torch
app = Flask(__name__)
from unet_segmentation.utils import get_masks, get_segmented_image, add_segment_to_image
import numpy as np
import cv2
from base64 import encodebytes
import io

alpha = 0.5
beta = 1.0 - alpha
model = torch.load('checkpoint/unet.pt')

@app.route("/upload", methods=["POST"])
def process_image():
    file1 = request.files['image']
    file2 = request.files['pattern']

    img = Image.open(file1.stream)
    pattern = Image.open(file2.stream)
    result = get_result(img, pattern)
    encoded_imges = []
    for out in result:
        encoded_imges.append(get_response_image(out))
    print("Results : " + str(len(encoded_imges)))
    return jsonify({'result': encoded_imges})

def get_result(img: Image, pattern: Image):
    pattern = pattern.resize(img.size)
    masks = get_masks(model, img)
    print("Created Masks")

    result = []
    for mask in masks:
        seg_img = get_segmented_image(img, mask)

        dst = cv2.addWeighted(np.array(seg_img), alpha, np.array(pattern), beta, 0.0)
        result.append(add_segment_to_image(img, dst, mask))
    return result

def get_response_image(image: np.ndarray):
    pil_img = Image.fromarray(image)
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

if __name__ == "__main__":
    app.run(debug=True)