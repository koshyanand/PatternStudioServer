import torch
from PIL import Image
from segmentation.utils import get_masks, get_segmented_image, add_segment_to_image
import matplotlib.pyplot as plt
import numpy as np
import cv2

model = torch.load('checkpoint/unet.pt')

image_path = "test_images/test.jpg"
pattern_path = "test_images/pattern.jpg"

img = Image.open(image_path)
pattern = Image.open(pattern_path).resize(img.size)
print("Loaded Images")
masks = get_masks(model, img)
print("Created Masks")

seg_img = get_segmented_image(img, masks[0])

alpha = 0.5
beta = 1.0 - alpha
dst = cv2.addWeighted(np.array(seg_img), alpha, np.array(pattern), beta, 0.0)


result = add_segment_to_image(img, dst, masks[0])

cv2.imwrite("output/out3.jpeg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))    

