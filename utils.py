import os

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize

def load_images(dir_path, resize_dim=(20, 20)):
    """Returns an array of flatten image arrays resized to `redize_dim`"""
    img_array = list()

    directory = os.fsencode(dir_path)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(dir_path, filename)
            image = imread(img_path)
            gray_image = rgb2gray(image)
            croped_image = resize(gray_image, resize_dim)
            img_array.append(croped_image.flatten())
    
    return img_array
