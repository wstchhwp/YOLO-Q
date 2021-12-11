from PIL import Image, ImageDraw
import numpy as np
from PIL import ImageFile
# import numbers

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_raito(new_size, original_size):
    """Get the ratio bewtten input_size and original_size"""
    # # mmdet way
    # iw, ih = new_size
    # ow, oh = original_size
    # max_long_edge = max(iw, ih)
    # max_short_edge = min(iw, ih)
    # ratio = min(max_long_edge / max(ow, oh), max_short_edge / min(ow, oh))
    # return ratio

    # # yolov5 way
    return min(new_size[0] / original_size[0], new_size[1] / original_size[1])

def imresize(img, new_size):
    """Resize the img with new_size by PIL(keep aspect).

    Args:
        img (PIL): The original image.
        new_size (tuple): The new size(w, h).
    """
    if isinstance(new_size, int):
        new_size = (new_size, new_size)
    old_size = img.size
    ratio = get_raito(new_size, old_size)
    img = img.resize((int(old_size[0] * ratio), int(old_size[1] * ratio)))
    return img

def get_wh(a, b):
    return np.random.randint(a, b)


def paste2(sample1, sample2, background, scale=1.2):
    sample1 = Image.open(sample1)
    d_w1, d_h1 = sample1.size

    sample2 = Image.open(sample2)
    d_w2, d_h2 = sample2.size

    # print(sample.size)
    background = Image.open(background)
    background = background.resize((int((d_w1 + d_w2) * scale), int((d_h1 + d_h2) * scale)))
    bw, bh = background.size

    x1, y1 = get_wh(0, int(d_w1 * scale) - d_w1), get_wh(0, bh - d_h1)
    x2, y2 = get_wh(int(d_w1 * scale), bw - d_w2), get_wh(0, bh - d_h2)
    # x1, y1 = get_wh(0, int(bw / 2) - d_w1), get_wh(0, bh - d_h1)
    # x2, y2 = get_wh(int(bw / 2), bw - d_w2), get_wh(0, bh - d_h2)

    background.paste(sample1, (x1, y1))
    background.paste(sample2, (x2, y2))
    # background = background.resize((416, 416))

    return np.array(background), (x1, y1, x2, y2), background
    # print(background.size)
    # background.show()


def paste1(sample, background, bg_size, fg_scale=1.5):
    sample = Image.open(sample)
    background = Image.open(background)
    background = imresize(background, bg_size)
    bw, bh = background.size
    # background = background.resize((int(d_w * scale), int(d_h * scale)))
    new_w, new_h = int(bw / fg_scale), int(bh / fg_scale)
    sample = imresize(sample, (new_w, new_h))

    d_w, d_h = sample.size
    x1, y1 = get_wh(0, bw - d_w), get_wh(0, bh - d_h)
    background.paste(sample, (x1, y1))
    # draw = ImageDraw.Draw(background)
    # draw.rectangle((x1 + 240, y1 + 254, x1 + 240 + 5, y1 + 254 + 5), 'red', 'green')
    # draw.rectangle((x1 + 80, y1 + 28, x1 + 400, y1 + 480), None, 'green')
    # background = background.resize((416, 416))

    return np.array(background.convert('RGB'))[:, :, ::-1], (x1, y1), background, (d_w, d_h)


if __name__ == '__main__':
    _, coord1, test1, _ = paste1('2007_000032.jpg', 'dark.jpg', bg_size=640, fg_scale=2)
    print(coord1)
    test1.show()
    # _, coord2, test2 = paste2('2007_000032.jpg', '0-9.jpg', 'dark.jpg', scale=2)
    # print(coord2)
    # test2.show()
