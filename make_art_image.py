import os
import cv2
import tqdm
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_text_background(ttf_path, text, font_size=512):
    font = ImageFont.truetype(ttf_path, font_size)
    text_width = font.getsize(text)

    image = Image.fromarray(np.zeros((text_width[1], text_width[0], 3), dtype=np.uint8))

    bg_size = image.size
    img_draw = ImageDraw.Draw(image)
    text_coordinate = int((bg_size[0] - text_width[0]) / 2), int((bg_size[1] - text_width[1]) / 2)
    img_draw.text(text_coordinate, text, font=font, fill=(255, 255, 255))

    # image.show()
    return np.array(image, dtype=np.uint8)


def resize_same_ratio(image, length=128):
    h, w, c = image.shape
    ratio = max(length / h, length / w)
    return cv2.resize(image, (int(round(w * ratio)), int(round(h * ratio))))


def crop_center(image, length, dim):
    size = image.shape
    offset = int((size[dim] - length) / 2)
    if dim == 0:
        return image[offset:offset + length]
    elif dim == 1:
        return image[:, offset:offset + length]
    else:
        return None


def cat_images(images_dir, out_size, cat_size=128):
    images = [cv2.imread(os.path.join(images_dir, x)) for x in tqdm.tqdm(os.listdir(images_dir), desc='reding...')]
    images = [resize_same_ratio(x, cat_size) for x in images]
    horizontal = [x for x in images if x.shape[0] < x.shape[1]]
    horizontal_min_w = min([x.shape[1] for x in horizontal])
    horizontal = [crop_center(x, horizontal_min_w, 1) for x in horizontal]
    vertical = [x for x in images if x.shape[0] > x.shape[1]]
    vertical_min_w = min([x.shape[0] for x in vertical])
    vertical = [crop_center(x, vertical_min_w, 0) for x in vertical]
    images = [vertical, horizontal]

    output = np.zeros(out_size, dtype=np.uint8)

    i = 0
    v_or_h = 0
    while i < out_size[1]:
        image_size = images[v_or_h][0].shape
        num_image = math.ceil(output.shape[0] / image_size[0])
        cat_images = random.sample(images[v_or_h], k=num_image)
        output[:, i:min(i + image_size[1], out_size[1])] = np.concatenate(cat_images, 0)[:out_size[0],
                                                           :min(image_size[1], out_size[1] - i)]

        i += image_size[1]
        v_or_h = abs(1 - v_or_h)

    return output


if __name__ == '__main__':
    ttf_path = r"C:\software\Font\ZhiMangXing-Regular.ttf"
    text = '王孙'
    mask = draw_text_background(ttf_path, text)

    images_dir = r"C:\Users\Yuxian\Downloads\hunsha"
    outsize = mask.shape
    image = cat_images(images_dir, outsize)

    image = cv2.addWeighted(image, 0.4, mask, 0.6, 1)

    cv2.imshow('play', image)
    cv2.waitKey()
