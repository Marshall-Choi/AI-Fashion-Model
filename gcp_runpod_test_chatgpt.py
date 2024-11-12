import json
import io
import time
import base64
import gradio as gr
import numpy as np
from PIL import Image
from openai import OpenAI
import asyncio
from api_request_parallel_processor import parallel_processor
import os

api_key = "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"
client = OpenAI(api_key=api_key)

import requests

RUNPOD_API_KEY = 'MJKX9ULY5CGPIRZ2P4SY7CZI4PSOM5N6OJOYD54M'  # Pion AI

RP_SERVER_ADDR = {'changebg': 'om4jeyohxkes2m',
                  'dyncraft': '467xw8410y4pn7',
                  'm2m': 'om4jeyohxkes2m',
                  'local' : '',
                  }

class RP_COMFY_Connector:
    def __init__(self, server_type):
        self.headers = {'Content-Type': 'application/json',
                        'accept': 'accept: application/json',
                        }
        self.server_type = server_type
        self.change_server(server_type)

    def change_server(self, server_type):
        if server_type == 'local':
            self.base_url = 'http://0.0.0.0:54310'
            self.server = 'local'

        elif server_type == 'runpod':
            self.base_url = 'https://api.runpod.ai/v2/'
            self.headers.update({'authorization': 'Bearer ' + RUNPOD_API_KEY})
            self.server = 'm2m'

        return True

    def submit_post_sync(self, server, data, timeout=60):
        ret = {}
        try:
            response = requests.post(f"{self.base_url}{RP_SERVER_ADDR.get(server)}/runsync",
                                     headers=self.headers,
                                     data=json.dumps(data).encode('utf-8'),
                                     timeout=(3, timeout))
            response.raise_for_status()
            response_body = response.json()
            
            comfyui_output_dict = response_body.get('output', {})
            
            if comfyui_output_dict.get('status') == 'error':
                print(f"An error occurred in ComfyUI: {comfyui_output_dict['message']}")
                for _details in comfyui_output_dict.get('details'):
                    print(_details, end='')
                
            elif comfyui_output_dict.get('status') == 'success':
                ret['outputs'] = comfyui_output_dict.get('outputs')
                ret['aux'] = comfyui_output_dict.get('aux')

            else:
                print("An error occurred: Unknown ComfyUI status")
                print(response_body)
                raise ValueError("An error occurred: Unknown ComfyUI status")
            return ret

        except requests.exceptions.Timeout:
            print(f"Request timed out")
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            print(f"{response.json()}")
        except requests.exceptions.RequestException as e:
            print(f"RequestExceptionerror occurred: {e}")
        except ValueError as e:
            print(e)
        except Exception as e:
            print("An generic error occurred:", e)
        return {}

    @staticmethod
    def convert_img_to_base64url(image, ext='webp'):
        buffered = io.BytesIO()
        image.save(buffered, format=ext, lossless=True)
        return f'data:image/{ext};base64,' + base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    def convert_base64url_to_img(img_url):
        return Image.open(io.BytesIO(base64.b64decode(img_url.split(",")[1])))

    @staticmethod
    def convert_vid_to_base64url(video, ext='mp4'):
        with open(video, 'rb') as video_file:
            video_data = video_file.read()
            return f'data:video/{ext};base64,' + base64.b64encode(video_data).decode("utf-8")

    @staticmethod
    def convert_base64url_to_vid(vid_url):
        vid_b64 = vid_url.split(",")[1]
        vid_bytes = base64.b64decode(vid_b64)
        video_path = f'/tmp/video{time.strftime("%Y%m%d_%H%M%S")}.mp4'
        with open(video_path, 'wb') as video_file:
            video_file.write(vid_bytes)
        return video_path

    def fix_request_param(self, cmd_str, images, **kwargs):
        image_urls = []
        for image in images:
            img_url = self.convert_img_to_base64url(image)
            image_urls.append(img_url)

        input_dict = {"command": cmd_str,
                      "image_urls": image_urls}
        input_dict.update(kwargs)

        return {"input": input_dict}

    def run_command(self, cmd_str, inputs, **kwargs):
        data = self.fix_request_param(cmd_str=cmd_str, images=inputs, **kwargs)
        gen_dict = self.submit_post_sync(server=self.server, data=data)
        return gen_dict
    
    # Generate Images

    def run_gen_upscale(self, img, skip_upscale):
        if skip_upscale:
            return {
                'image' : img
            }
        gen_dict = self.run_command(cmd_str='upscale', inputs=[img])
        return {
            'image' : self.convert_base64url_to_img(img_url=gen_dict.get('outputs')[0].get('url'))
        }

    def run_rembg(self, img):
        gen_dict = self.run_command(cmd_str='rmbg', inputs=[img])
        return {
            'image' : self.convert_base64url_to_img(img_url=gen_dict.get('outputs')[0].get('url'))
        }

    def run_gen_depth(self, img):
        gen_dict = self.run_command(cmd_str='depth', inputs=[img])
        return {
            'image' : self.convert_base64url_to_img(img_url=gen_dict.get('outputs')[0].get('url'))
        }

    def run_garment_detector(self, img, image_selection):
        gen_dict = self.run_command(cmd_str='GenModel_detection', inputs=[img], prompt=image_selection)
        if image_selection == "":
            return {
                'image' : self.convert_base64url_to_img(img_url=gen_dict.get('outputs')[0].get('url')),
                'index' : gen_dict.get('aux')
            }
        else:
            return {
                'image' : self.convert_base64url_to_img(img_url=gen_dict.get('outputs')[0].get('url'))
            }
    
    def run_whiteBG_garment(self, img, mask):
        gen_dict = self.run_command(cmd_str='GenModel_whiteBG_garment', inputs=[img, mask])
        return {
            'image' : self.convert_base64url_to_img(img_url=gen_dict.get('outputs')[0].get('url'))
        }

    def run_gen_model(self, img_fashion_upscale, img_fashion_whiteBG_garment, garment_mask, prompt):
        gen_dict = self.run_command(cmd_str='GenModel_genmodel', inputs=[img_fashion_upscale, img_fashion_whiteBG_garment, garment_mask], prompt=prompt)
        return {
            'image' : self.convert_base64url_to_img(img_url=gen_dict.get('outputs')[0].get('url'))
        }

    def run_whiteBG_model(self, model_img):
        gen_dict = self.run_command(cmd_str='GenModel_whiteBG_model', inputs=[model_img])
        return {
            'image' : self.convert_base64url_to_img(img_url=gen_dict.get('outputs')[0].get('url'))
        }

    def run_face_detailer(self, img_fashion_whiteBG_model, img_fashion_whiteBG_garment, garment_mask, prompt):
        gen_dict = self.run_command(cmd_str='GenModel_detailer', inputs=[img_fashion_whiteBG_model, img_fashion_whiteBG_garment, garment_mask], prompt=prompt)
        return {
            'image' : self.convert_base64url_to_img(img_url=gen_dict.get('outputs')[0].get('url'))
        }

    def run_gen_BG(self, masked_bg_img, img_fashion_whiteBG_garment, img_fashion_garment_mask, prompt, restoration_choices, lora, lora_strength, clip_skip):
        lora_setting = {
            'lora' : lora,
            'strength' : lora_strength,
        }
        gen_dict = self.run_command(cmd_str='GenModel_genBG', inputs=[masked_bg_img, img_fashion_whiteBG_garment, img_fashion_garment_mask], prompt=prompt, restoration_choices=restoration_choices, lora_setting=lora_setting, clip_skip=clip_skip)
        return {
            'images' : [(self.convert_base64url_to_img(image.get('url')), image.get('restoration_choice')) for image in gen_dict.get('outputs')]
        }

    def run_segmentation(self, img):
        gen_dict = self.run_command(cmd_str='GenModel_segmentation', inputs=[img])
        return {
            'images' : [self.convert_base64url_to_img(image.get('url')) for image in gen_dict.get('outputs')]
        }

    def run_preview_mask(self, img, mask):
        gen_dict = self.run_command(cmd_str='GenModel_preview_mask', inputs=[img, mask])
        return {
            'image' : self.convert_base64url_to_img(img_url=gen_dict.get('outputs')[0].get('url'))
        }

    def run_postprocess_mask(self, mask):
        gen_dict = self.run_command(cmd_str='GenModel_grow_mask', inputs=[mask])
        return {
            'image' : self.convert_base64url_to_img(img_url=gen_dict.get('outputs')[0].get('url'))
        }

    def run_auto_detection(self, img, item):
        gen_dict = self.run_command(cmd_str='GenModel_auto_detection', inputs=[img], fashion_item=item)
        return {
            'images' : [(self.convert_base64url_to_img(image.get('url'))) for image in gen_dict.get('outputs')]
        }


rpcomfy = RP_COMFY_Connector('local')


# Functions using ComfyUI


def run_fashion_gen_upscale(img, skip_upscale):
    return rpcomfy.run_gen_upscale(img=img, skip_upscale=skip_upscale).get('image')

def run_fashion_whiteBG_garment(img, mask):
    return rpcomfy.run_whiteBG_garment(img=img, mask=mask).get('image')

def run_fashion_gen_model(img_fashion_upscale, img_fashion_whiteBG_garment, garment_mask, face_prompt):
    return rpcomfy.run_gen_model(img_fashion_upscale=img_fashion_upscale, img_fashion_whiteBG_garment=img_fashion_whiteBG_garment, garment_mask=garment_mask, prompt=face_prompt).get('image')

def run_fashion_whiteBG_model(model_img):
    return rpcomfy.run_whiteBG_model(model_img=model_img).get('image')

def run_fashion_face_detailer(img_fashion_whiteBG_model, img_fashion_whiteBG_garment, garment_mask, prompt):
    return rpcomfy.run_face_detailer(img_fashion_whiteBG_model=img_fashion_whiteBG_model, img_fashion_whiteBG_garment=img_fashion_whiteBG_garment, garment_mask=garment_mask, prompt=prompt).get('image')

def run_fashion_change_BG(masked_bg_img, img_fashion_whiteBG_garment, img_fashion_garment_mask, prompt, restoration_choices, lora, lora_strength, clip_skip):
    return rpcomfy.run_gen_BG(masked_bg_img=masked_bg_img, img_fashion_whiteBG_garment=img_fashion_whiteBG_garment, img_fashion_garment_mask=img_fashion_garment_mask, prompt=prompt, restoration_choices=restoration_choices, lora=lora, lora_strength=lora_strength, clip_skip=clip_skip).get('images')

def run_fashion_segmentation(img):
    return rpcomfy.run_segmentation(img=img).get('images')

def run_fashion_preview_mask(img, mask):
    return rpcomfy.run_preview_mask(img=img, mask=mask).get('image')

def run_fashion_postprocess_mask(mask):
    return rpcomfy.run_postprocess_mask(mask=mask).get('image')

def run_fashion_garment_detector(img):
    obj_list = []
    num_objects = rpcomfy.run_garment_detector(img=img, image_selection="").get('index')
    for i in range(num_objects):
        detected_obj = rpcomfy.run_garment_detector(img=img, image_selection=str(i)).get('image')
        obj_list.append(detected_obj)
    return obj_list

def run_fashion_get_segments(img):
    detected_masks = run_fashion_garment_detector(img=img)
    segmented_masks = run_fashion_segmentation(img=img)
    total_mask_list = detected_masks + segmented_masks
    return total_mask_list

def run_fashion_auto_detection(img):
    base64_image = convert_img_to_base64url(img)
    message_for_item_list = "Give all the bboxes where the mannequin skin is showing"
    item = chatgpt_api_single(base64_image, message_for_item_list)
    img_detected_items = rpcomfy.run_auto_detection(img=img, item=item).get('images')[1]

    # input coordinates and get mask
    return img_detected_items


# Utility Functions


def get_combined_mask(segmented_masks, indices):
    def load_mask(mask_path):
        return Image.open(mask_path).convert("L")

    base_mask = Image.new("L", load_mask(segmented_masks[0][0]).size, color=0)

    if not indices:  
        return base_mask

    indices = [int(i) - 1 for i in indices.split(',')]

    for index in indices:
        mask_image = load_mask(segmented_masks[index][0])
        base_mask = Image.composite(mask_image, base_mask, mask_image)

    return base_mask

clicked_indices = []

def create_annotated_image(base_image, segments):
    clicked_indices.clear()
    annotations = []
    base_image = base_image.convert("RGBA")

    if segments is None:
        return base_image, annotations

    for idx, segment in enumerate(segments, start=1):
        segment_image = Image.open(segment[0]).convert("L")
        binary_segment = np.where(np.array(segment_image) > 128, 1, 0)
        annotations.append((binary_segment, str(idx)))

    return base_image, annotations

def on_annotation_click(evt: gr.SelectData):  
    index = evt.index + 1 

    if index not in clicked_indices:
        clicked_indices.append(index)

    return ','.join(map(str, clicked_indices))

def toggle_interface(current_state):
    return not current_state

def chatgpt_api_single(base64_image, msg):
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
        "role": "user",
        "content": [
            {"type": "text", "text": f"{msg}"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"{base64_image}",
                    "detail": "low"
                },
            },
        ],
        }
    ],
    max_tokens=300,
    )

    print(response.choices[0].message.content)
    return 
    #Beige blazer, Black T-shirt, Blue jeans, Gray socks, White sneakers
    #Black cap, Black sleeveless dress, Plaid shirt tied around waist, Black backpack, Gold bracelets

def create_jsonl_file(base64_images, output_filepath, model="gpt-4o-mini", max_tokens=300):
    idx = 1  
    with open(output_filepath, "w") as f:
        for base64_image in enumerate(base64_images):
            data = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{idx}: Please analyse this image. Respond with '1' if the masked area is only composed of wearables. Otherwise, if the masked area contains anything that is not clothing or accessories, respond with '0"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"{base64_image[1]}",
                                    "detail": "low"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": max_tokens,
            }

            f.write(json.dumps(data) + "\n")
            idx += 1

def convert_img_to_base64url(image, ext='jpeg'):
    buffered = io.BytesIO()
    image.save(buffered, format=ext)
    return f'data:image/{ext};base64,' + base64.b64encode(buffered.getvalue()).decode("utf-8")

def preprocess_image_for_api(img, masks):
    idx = 0
    preprocessed_imgs = []
    for mask in masks:
        mask_image = Image.open(mask[0]).convert('L')
        composite_image = Image.composite(img, Image.new("RGB", img.size, (255, 255, 255)), mask_image)
        
        base64_image = convert_img_to_base64url(composite_image)
        preprocessed_imgs.append(base64_image)
        
        # 파일로 저장
        #file_path = f'processed_image_{idx}.jpg'  # 또는 원하는 확장자 사용
        #composite_image.save(file_path)  # 이미지 저장
        #idx += 1  # idx 증가

    return preprocessed_imgs

def preprocess_image_for_api1(img, masks):
    preprocessed_imgs = []

    mask_color = (100, 15, 240)  
    mask_opacity = 0.61  
    idx = 1

    for mask_tuple in masks:
        mask_path = mask_tuple[0] 

        mask_image = Image.open(mask_path).convert('L')  
        mask_array = np.array(mask_image) / 255.0  

        mask_adjusted = mask_array * mask_opacity

        img_rgba = img.convert("RGBA")

        overlay_image = Image.new("RGBA", img.size, mask_color + (int(255 * mask_opacity),))

        mask_composite = Image.fromarray((mask_adjusted * 255).astype(np.uint8), mode='L')
        composite_image = Image.composite(overlay_image, img_rgba, mask_composite)

        composite_image = composite_image.convert("RGB")
        base64_image = convert_img_to_base64url(composite_image)
        preprocessed_imgs.append(base64_image)

        file_path = f'processed_image_{idx}.jpg'  # 또는 원하는 확장자 사용
        composite_image.save(file_path)  # 이미지 저장
        idx += 1

    return preprocessed_imgs

def mask_classifier(img, masks):
    input_imgs = preprocess_image_for_api1(img=img, masks=masks)
    create_jsonl_file(base64_images=input_imgs, output_filepath="requests.jsonl")

    parallel_processor(requests_filepath="requests.jsonl", save_filepath="result.jsonl", api_key=client.api_key)

    indices = []
    tokens_used = 0

    with open('result.jsonl', 'r') as file:
        for line in file:
            obj = json.loads(line.strip()) 
            result = obj[1].get("choices")[0].get("message").get("content") 
            tokens_used += obj[1].get("usage").get("total_tokens")
            if result == '1':
                index = obj[0].get("messages")[0].get("content")[0].get("text").split(':')[0]
                indices.append(index)

    print("Result:", indices)
    print("Used", tokens_used, "tokens")

    global clicked_indices
    clicked_indices.clear()
    clicked_indices += indices
    result_string = ','.join(indices)

    if os.path.exists("requests.jsonl"):
        os.remove("requests.jsonl")

    if os.path.exists("result.jsonl"):
        os.remove("result.jsonl")

    return result_string


# Gradio Interface


with gr.Blocks(analytics_enabled=False, title='imgthis') as demo5:
    with gr.Row():
        gr.Markdown("## AI Model Generation")

    # Image Input
    
    with gr.Row():
        img_fashion_input = gr.Image(type='pil', image_mode="RGB")

    # Image Preprocess

    with gr.Group():
        with gr.Row():
            btn_fashion_skip_upscale = gr.Checkbox(label="Skip Upscale", value=False)
            btn_fashion_preprocess = gr.Button("Preprocess Image", variant="primary")
        with gr.Row():
            img_fashion_upscale = gr.Image(type='pil', label='upscale', image_mode='RGB')
            img_fashion_whiteBG = gr.Image(type='pil', label='whiteBG', image_mode='RGB')
    
    # RMBG Retrial

    fashion_state_show_rmbg_retrial = gr.State(False)
    btn_fashion_show_rmbg_retrial = gr.Button("Retry RMBG")
    
    with gr.Group(visible=False) as rmbg_retrial_interface:
        with gr.Row():
            btn_fashion_segmentation_rmbg_retry = gr.Button("Retry RMBG(wait after clicking)", variant="primary")
            fashion_segment_indices_rmbg_retry = gr.Textbox(label="Select Segments(Input numbers or Click the annotations)")
            btn_fashion_preview_rmbg_retry = gr.Button("View New RMBG", variant="primary")
        with gr.Row():
            img_fashion_segments_rmbg_retry = gr.Gallery(label='RMBG Segments', columns=[5], visible=False)
        with gr.Row():
            img_fashion_annotated_rmbg_retry = gr.AnnotatedImage(label="Annotated Image")
            img_fashion_combined_mask_rmbg_retry = gr.Image(type='pil', label='mask', image_mode='RGBA', visible=False)
            img_fashion_postprocessed_mask_rmbg_retry = gr.Image(type='pil', label='mask', image_mode='RGBA', visible=False)
            img_fashion_preview_rmbg_retry = gr.Image(type='pil', label='New RMBG', image_mode='RGBA')

    # Image Auto Mask

    with gr.Group():
        with gr.Row():
            btn_fashion_auto_mask = gr.Button("View Auto Mask", variant="primary")
        with gr.Row():
            img_fashion_auto_mask = gr.Image(type='pil', label='auto mask', image_mode='RGBA')
            img_fashion_auto_mask_preview = gr.Image(type='pil', label='auto mask preview', image_mode='RGBA')

    # Image Segmentation

    with gr.Group():
        with gr.Row():
            btn_fashion_segmentation = gr.Button("View Segments(wait after clicking)", variant="primary")
            fashion_segment_indices = gr.Textbox(label="Select Segments(Input numbers or Click the annotations)")
            btn_fashion_mask_preview = gr.Button("View Mask", variant="primary")
        with gr.Row():
            img_fashion_segments = gr.Gallery(label='segment masks', columns=[5], visible=True)
        with gr.Row():
            img_fashion_annotated_image_for_mask = gr.AnnotatedImage(label="Annotated Image")
            img_fashion_combined_mask = gr.Image(type='pil', label='combined mask', image_mode='RGBA', visible=False)
            img_fashion_combined_mask_preview = gr.Image(type='pil', label='combined mask preview', image_mode='RGBA')
            img_fashion_garment_mask = gr.Image(type='pil', label='garment mask', image_mode='RGBA', visible=False)
            img_fashion_whiteBG_garment = gr.Image(type='pil', label='whiteBG_garment', image_mode='RGBA', visible=True)
            
    # Model Generation

    with gr.Group():
        with gr.Row():
            fashion_face_prompt = gr.Textbox(label="Face Prompt")
            btn_fashion_model_gen = gr.Button("Gen Face", variant="primary")
        with gr.Row():
            img_fashion_model = gr.Image(type='pil', label='model', image_mode='RGB')
            img_fashion_whiteBG_model = gr.Image(type='pil', label='whiteBG_model', image_mode='RGB')
            img_fashion_with_details = gr.Image(type='pil', label='detail', image_mode='RGB')

    # BG Generation

    with gr.Group():
        with gr.Row():
            fashion_bg_text = gr.Textbox(label='Background prompt')
            fashion_restoration_choices = gr.CheckboxGroup(['1', '2', '3', '4', '5', '6'], label="Restoration", value=['1', '2', '3', '4', '5', '6'])
            fashion_lora = gr.Radio(['v1', 'v2', 'Not use'], value='v1', label="LoRA")
            fashion_lora_strength = gr.Slider(0, 1, value=0.90, step=0.01, label="LoRA Strength")
            fashion_clip_skip = gr.Checkbox(label="Clip Skip", value=True)

        with gr.Row():
            btn_fashion_bgic = gr.Button(f"ChangeBG", variant="primary")

        with gr.Row():
            img_fashion_changbg_gallery = gr.Gallery(label="Generated Images", columns=[5], rows=[1], object_fit="contain", height="auto")


    # Control Buttons


    # Image Preprocess

    btn_fashion_preprocess.click(fn=run_fashion_gen_upscale, inputs=[img_fashion_input, btn_fashion_skip_upscale], outputs=img_fashion_upscale
                            ).then(fn=run_fashion_whiteBG_model, inputs=[img_fashion_upscale], outputs=img_fashion_whiteBG)
    
    # RMBG Retrial
    
    btn_fashion_show_rmbg_retrial.click(toggle_interface, inputs=[fashion_state_show_rmbg_retrial], outputs=[fashion_state_show_rmbg_retrial], show_progress=False)
    
    fashion_state_show_rmbg_retrial.change(lambda visible: gr.update(visible=visible), inputs=[fashion_state_show_rmbg_retrial], outputs=rmbg_retrial_interface)

    btn_fashion_segmentation_rmbg_retry.click(fn=run_fashion_segmentation, inputs=[img_fashion_whiteBG], outputs=img_fashion_segments_rmbg_retry
                            ).then(fn=create_annotated_image, inputs=[img_fashion_whiteBG, img_fashion_segments_rmbg_retry], outputs=img_fashion_annotated_rmbg_retry)

    img_fashion_annotated_rmbg_retry.select(fn=on_annotation_click, inputs=[], outputs=[fashion_segment_indices_rmbg_retry])

    btn_fashion_preview_rmbg_retry.click(fn=get_combined_mask, inputs=[img_fashion_segments_rmbg_retry, fashion_segment_indices_rmbg_retry], outputs=img_fashion_combined_mask_rmbg_retry
                            ).then(fn=run_fashion_preview_mask, inputs=[img_fashion_whiteBG, img_fashion_combined_mask_rmbg_retry], outputs=img_fashion_preview_rmbg_retry
                            ).then(fn=run_fashion_postprocess_mask, inputs=img_fashion_combined_mask_rmbg_retry, outputs=img_fashion_postprocessed_mask_rmbg_retry
                            ).then(fn=run_fashion_whiteBG_garment, inputs=[img_fashion_whiteBG, img_fashion_combined_mask_rmbg_retry], outputs=[img_fashion_whiteBG])

    # Image Auto Mask
    btn_fashion_auto_mask.click(fn=run_fashion_auto_detection, inputs=[img_fashion_whiteBG], outputs=img_fashion_auto_mask
                            ).then(fn=run_fashion_preview_mask, inputs=[img_fashion_whiteBG, img_fashion_auto_mask], outputs=img_fashion_auto_mask_preview)

    # Image Segmentation

    btn_fashion_segmentation.click(fn=run_fashion_get_segments, inputs=[img_fashion_whiteBG], outputs=img_fashion_segments
                            ).then(fn=create_annotated_image, inputs=[img_fashion_whiteBG, img_fashion_segments], outputs=img_fashion_annotated_image_for_mask
                            ).then(fn=mask_classifier, inputs=[img_fashion_whiteBG, img_fashion_segments], outputs=[fashion_segment_indices]
                            ).then(fn=get_combined_mask, inputs=[img_fashion_segments, fashion_segment_indices], outputs=img_fashion_combined_mask
                            ).then(fn=run_fashion_preview_mask, inputs=[img_fashion_whiteBG, img_fashion_combined_mask], outputs=img_fashion_combined_mask_preview)

    img_fashion_annotated_image_for_mask.select(fn=on_annotation_click, inputs=[], outputs=[fashion_segment_indices])

    btn_fashion_mask_preview.click(fn=get_combined_mask, inputs=[img_fashion_segments, fashion_segment_indices], outputs=img_fashion_combined_mask
                            ).then(fn=run_fashion_preview_mask, inputs=[img_fashion_whiteBG, img_fashion_combined_mask], outputs=img_fashion_combined_mask_preview
                            ).then(fn=run_fashion_postprocess_mask, inputs=img_fashion_combined_mask, outputs=img_fashion_garment_mask
                            ).then(fn=run_fashion_whiteBG_garment, inputs=[img_fashion_whiteBG, img_fashion_garment_mask], outputs=img_fashion_whiteBG_garment)
    
    # Model Generation

    btn_fashion_model_gen.click(fn=run_fashion_gen_model, inputs=[img_fashion_whiteBG, img_fashion_whiteBG_garment, img_fashion_garment_mask, fashion_face_prompt], outputs=img_fashion_model
                            ).then(fn=run_fashion_whiteBG_model, inputs=[img_fashion_model], outputs=img_fashion_whiteBG_model
                            ).then(fn=run_fashion_face_detailer, inputs=[img_fashion_whiteBG_model, img_fashion_upscale, img_fashion_garment_mask, fashion_face_prompt], outputs=img_fashion_with_details)

    # BG Generation

    btn_fashion_bgic.click(fn=run_fashion_change_BG, inputs=[img_fashion_with_details, img_fashion_upscale, img_fashion_garment_mask, fashion_bg_text, fashion_restoration_choices, fashion_lora, fashion_lora_strength, fashion_clip_skip], outputs=img_fashion_changbg_gallery)

demo5.launch(server_name='0.0.0.0', server_port=54309)
