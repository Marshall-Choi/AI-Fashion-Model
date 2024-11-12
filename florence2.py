import copy
import datetime
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import requests
import sys
import torch

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM

model_id = 'microsoft/Florence-2-large'

current_dir = os.getcwd()  # 현재 디렉토리 경로 가져오기
image_path = os.path.join(current_dir, "yolo_finetune", "whitebg_mannequin.png")  # 'yolofinetune' 폴더 내의 이미지 경로

# 이미지 열기
image = Image.open(image_path)

model = AutoModelForCausalLM.from_pretrained(model_id,
                                             trust_remote_code=True).eval()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def florence2(task_prompt, text_input=None):
    """
    Calling the Microsoft Florence2 model
    """
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt")

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids,
                                            skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height))

    return parsed_answer

def draw_polygons(image, prediction, fill_mask=False):
    """
    Draws segmentation masks with polygons on an image.

    Parameters:
    - image_path: Path to the image file.
    - prediction: Dictionary containing 'polygons' and 'labels' keys.
                  'polygons' is a list of lists, each containing vertices of a polygon.
                  'labels' is a list of labels corresponding to each polygon.
    - fill_mask: Boolean indicating whether to fill the polygons with color.
    """
    draw = ImageDraw.Draw(image)
    scale = 1

    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = "lime"
        fill_color = "lime" if fill_mask else None

        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue

            _polygon = (_polygon * scale).reshape(-1).tolist()
            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=fill_color)
            else:
                draw.polygon(_polygon, outline=color)
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)

    image.save("output_image.png")

def draw_bbox_and_save(image, data):
    draw = ImageDraw.Draw(image)
    
    try:
        # 폰트 설정 (기본 시스템 폰트 사용)
        font = ImageFont.load_default()
    except IOError:
        font = None

    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox

        # Bounding box 그리기
        draw.rectangle([(x1, y1), (x2, y2)], outline='lime', width=2)

        # 라벨 텍스트 추가
        if font:
            text_size = font.getsize(label)
        else:
            text_size = (len(label) * 6, 10)  # 임시 텍스트 크기

        text_position = (x1, y1 - text_size[1] if y1 - text_size[1] > 0 else y1)

        draw.rectangle([text_position, (x1 + text_size[0], y1)], fill='lime')
        draw.text((x1, y1 - text_size[1]), label, fill='black', font=font)

    # 이미지 저장
    image.save("output.jpg")

output_image = copy.deepcopy(image)

task_prompt = '<REFERRING_EXPRESSION_SEGMENTATION>'
results = florence2(task_prompt, text_input="all clothings and accessories that the mannequin is wearing")

draw_polygons(output_image,
              results['<REFERRING_EXPRESSION_SEGMENTATION>'],
              fill_mask=True)