from PIL import Image

import torch
from torchvision import transforms
import numpy as np
from unet_segmentation.unet import Unet
from unet_segmentation.post_processing import (
    prediction_to_classes,
    mask_from_prediction,
    remove_predictions
)

DEEP_FASHION2_CUSTOM_CLASS_MAP = {
    1: "trousers",
    2: "skirt",
    3: "top",
    4: "dress",
    5: "outwear",
    6: "shorts"
}
def get_masks(
    unet: Unet,
    img: Image,
    image_resize: int = 512,
    label_map: dict = DEEP_FASHION2_CUSTOM_CLASS_MAP,
    device: str = 'cuda',
    min_area_rate: float = 0.05) -> None:
    
    # Load image tensor
    img_tensor = _preprocess_image(img, image_size=image_resize).to(device)

    # Predict classes from model
    prediction_map = \
        torch.argmax(unet(img_tensor), dim=1).squeeze(0).cpu().numpy()

    # Remove spurious classes
    classes = prediction_to_classes(prediction_map, label_map)
    predicted_classes = list(
        filter(lambda x: x.area_ratio >= min_area_rate, classes))
    spurious_classes = list(
        filter(lambda x: x.area_ratio < min_area_rate, classes))
    clean_prediction_map = remove_predictions(spurious_classes, prediction_map)

    # Get masks for each of the predictions
    masks = [
        mask_from_prediction(predicted_class, clean_prediction_map)
        for predicted_class in predicted_classes
    ]

    result = []
    for mask in masks:
        result.append(mask.resize(img).binary_mask)
    return result

def _preprocess_image(image: Image, image_size: int) -> torch.Tensor:
    preprocess_pipeline = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0,), std=(1,))
    ])
    return preprocess_pipeline(image).unsqueeze(0)

def get_segmented_image(img: Image, mask: np.ndarray):
    img_pix = np.array(img)
    idx=(mask==0)
    img_pix[idx] = 0
    return img_pix

def add_segment_to_image(img: Image, blended_img: np.ndarray, mask: np.ndarray):
    img_pix = np.array(img)
    idx=(mask==0)
    blended_img[idx] = img_pix[idx]
    return blended_img
