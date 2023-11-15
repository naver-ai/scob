"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

import glob
from random import random, randrange

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def rand_color(_min=0, _max=150):
    return (randrange(_min, _max), randrange(_min, _max), randrange(_min, _max))


class Online_renderer:
    def __init__(
        self,
        max_img_w=768,
        max_img_h=768,
        min_img_w=400,
        min_img_h=400,
        bg_rgb_min=151,
        is_char=False,
        max_seq_len=512,
        blur_ratio=0.2,
    ):

        self.font_filepaths = glob.glob("./fonts/**/*.ttf", recursive=True)

        self.max_img_w = max_img_w
        self.max_img_h = max_img_h
        self.min_img_w = min_img_w
        self.min_img_h = min_img_h

        self.bg_rgb_min = bg_rgb_min
        self.is_char = is_char
        self.max_seq_len = max_seq_len
        self.blur_ratio = blur_ratio

    def gen_img(self, words):
        img_w = randrange(self.min_img_w, self.max_img_w)
        img_h = randrange(self.min_img_h, self.max_img_h)

        img = Image.new(
            "RGB",
            (img_w, img_h),
            rand_color(self.bg_rgb_min, 255),
        )

        draw = ImageDraw.Draw(img)
        quads = []

        box_len = 0 if len(words) < 29 else min(len(words), 60)
        max_font_size = int(41 - 30 * (box_len / 60))

        seq_len = 4
        words_not_blank = []
        otor_words_idx = []
        for words_idx, word in enumerate(words):
            font_idx, font_size = randrange(0, len(self.font_filepaths)), randrange(
                10, max_font_size
            )
            font = ImageFont.truetype(self.font_filepaths[font_idx], size=font_size)
            text = word["text"]
            text = text.replace("[UNK]", " ").strip()
            if text == "":
                continue
            elif text == "[DONTCARE]":
                continue
            elif text == "###":
                continue
            seq_len += 6 + len(text)
            if seq_len > self.max_seq_len:
                break
            words_not_blank.append({"text": text})
            otor_words_idx.append(words_idx)

            font_width, font_height = font.getbbox(text)[-2:]
            xy_cord = (
                randrange(0, max(img_w - font_width, 1)),
                randrange(0, img_h - font_height),
            )

            stroke_width = 0 if random() < 0.8 else randrange(1, 3)
            draw.text(
                xy_cord,
                text,
                font=font,
                fill=rand_color(),
                stroke_width=stroke_width,
                # stroke_fill=rand_color(0, 255),
            )

            if self.is_char:
                for i, char in enumerate(text):
                    bottom_1 = font.getsize(text[i])[1]
                    right, bottom_2 = font.getsize(text[: i + 1])
                    bottom = bottom_1 if bottom_1 < bottom_2 else bottom_2
                    width, height = font.getmask(char).size
                    right += xy_cord[0]
                    bottom += xy_cord[1]
                    top = bottom - height
                    left = right - width
                    quad = [[left, top], [right, top], [right, bottom], [left, bottom]]
                    quads.append(quad)
            else:
                left, top = xy_cord
                right = left + font_width
                bottom = top + font_height
                quad = [[left, top], [right, top], [right, bottom], [left, bottom]]
                quads.append(quad)

        if random() < self.blur_ratio:
            img = img.filter(ImageFilter.GaussianBlur(radius=(random() * 1.8)))

        img = np.array(img, dtype=np.uint8)
        quads = np.array(quads, dtype=np.float32)

        return img, quads, words_not_blank, otor_words_idx
