import json
import io
import time
import base64
import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import requests

RUNPOD_API_KEY = 'MJKX9ULY5CGPIRZ2P4SY7CZI4PSOM5N6OJOYD54M'  # Pion AI

RP_SERVER_ADDR = {'changebg': 'yazpphnxvussu9',
                  'dyncraft': '467xw8410y4pn7',
                  'm2m': '4dc8kcjhinteox',
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

    def submit_post_sync(self, server, data, timeout=120):
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

    def run_command(self, server, inputs, **kwargs):
        if self.server == 'local':
            server = 'local' 
        data = self.fix_request_param(cmd_str='change_model', images=inputs, **kwargs)
        gen_dict = self.submit_post_sync(server=server, data=data)
        return gen_dict
    
    # Generate Images

    def run_gen_upscale(self, img):
        gen_dict = self.run_command(server='m2m', inputs=[img], workflow_select='upscale')
        return {
            'image' : self.convert_base64url_to_img(img_url=gen_dict.get('outputs')[0].get('url'))
        }

    def run_garment_detector(self, img):
        gen_dict = self.run_command(server='m2m', inputs=[img], workflow_select='detection')
        return {
            'images' : [self.convert_base64url_to_img(image.get('url')) for image in gen_dict.get('outputs')]
        }

    def run_white_bg_model(self, model_img):
        gen_dict = self.run_command(server='m2m', inputs=[model_img], workflow_select='whiteBG_model')
        return {
            'image' : self.convert_base64url_to_img(img_url=gen_dict.get('outputs')[0].get('url'))
        }

    def run_segmentation(self, img):
        gen_dict = self.run_command(server='m2m', inputs=[img], workflow_select='segmentation')
        return {
            'images' : [self.convert_base64url_to_img(image.get('url')) for image in gen_dict.get('outputs')]
        }

    def run_gen_model_bg_at_once(self, img_whitebg, model_prompt, bool_auto_mask, bool_face_detailer, prompt, lora, lora_strength, img_garment_mask=None):
        lora_setting = {
            'lora' : lora,
            'strength' : lora_strength,
        }
        if img_garment_mask == None:
            bool_auto_mask = True
            gen_dict = self.run_command(server='m2m', inputs=[img_whitebg], workflow_select='genmodel', model_prompt=model_prompt, use_automask=bool_auto_mask, use_fd=bool_face_detailer, prompt=prompt, lora_setting=lora_setting)
        else:
            gen_dict = self.run_command(server='m2m', inputs=[img_whitebg, img_garment_mask], workflow_select='genmodel', model_prompt=model_prompt, use_automask=bool_auto_mask, use_fd=bool_face_detailer, prompt=prompt, lora_setting=lora_setting)
        return {
            'images' : [(self.convert_base64url_to_img(image.get('url')), image.get('restoration_choice')) for image in gen_dict.get('outputs')]
        }


# Control


class Control:
    def __init__(self, server):
        self.rpcomfy = RP_COMFY_Connector(server)

    def run_fashion_gen_upscale(self, img):
        return self.rpcomfy.run_gen_upscale(img=img)

    def run_fashion_make_white_bg_model(self, model_img):
        return self.rpcomfy.run_white_bg_model(model_img=model_img)

    def run_fashion_segmentation(self, img):
        return self.rpcomfy.run_segmentation(img=img)

    def run_fashion_garment_detector(self, img):
        return self.rpcomfy.run_garment_detector(img=img).get('images')

    def run_fashion_get_segments(self, img, bool_use_detection):
        detected_masks = []
        if bool_use_detection:
            detected_masks = self.run_fashion_garment_detector(img=img)
        segmented_masks = self.run_fashion_segmentation(img=img).get('images')
        total_mask_list = detected_masks + segmented_masks
        return total_mask_list

    def get_combined_mask(self, segmented_masks, indices, bool_use_auto_detected_mask=None, auto_detected_mask=None):
        if not bool_use_auto_detected_mask or auto_detected_mask == None:
            base_mask = Image.new("L", Image.open(segmented_masks[0]).size, color=0)
        else:
            base_mask = auto_detected_mask

        if not indices:
            return base_mask

        indices = [int(i) - 1 for i in indices.split(',')]

        for index in indices:
            mask_image = Image.open(segmented_masks[index]).convert("L")
            base_mask = Image.composite(mask_image, base_mask, mask_image)

        return base_mask

    def run_fashion_make_white_bg(self, image, mask):
        image = image.convert("RGBA")
        mask = mask.convert("L")
        inverted_mask = ImageOps.invert(mask)
        white_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        masked_image = Image.composite(image, white_bg, inverted_mask)
        return masked_image

    def run_fashion_preview_mask(self, img, mask):
        mask_color = (100, 15, 240)
        mask_opacity = 0.61

        img_rgba = img.convert("RGBA")

        mask_image = mask.convert('L')
        mask_array = np.array(mask_image) / 255.0

        mask_adjusted = mask_array * mask_opacity

        overlay_image = Image.new("RGBA", img.size, mask_color + (int(255 * mask_opacity),))

        mask_composite = Image.fromarray((mask_adjusted * 255).astype(np.uint8), mode='L')
        preview_img = Image.composite(overlay_image, img_rgba, mask_composite)

        preview_img = preview_img.convert("RGB")

        return preview_img

    def run_fashion_model_bg_at_once(self, img_whitebg, img_garment_mask, model_prompt, bool_auto_mask, bool_face_detailer, prompt, lora, lora_strength):
        return self.rpcomfy.run_gen_model_bg_at_once(
            img_whitebg=img_whitebg,
            img_garment_mask=img_garment_mask,
            model_prompt=model_prompt,
            bool_auto_mask=bool_auto_mask,
            bool_face_detailer=bool_face_detailer,
            prompt=prompt,
            lora=lora,
            lora_strength=lora_strength
        )


# View


cc = Control('local')


# Fashion  Begin


def run_fashion_gen_upscale(img, bool_upscale):
    return cc.run_fashion_gen_upscale(img=img).get('image') if bool_upscale else img


def run_fashion_make_white_bg_model(model_img):
    return cc.run_fashion_make_white_bg_model(model_img=model_img).get('image')


def run_fashion_change_model_bg_at_once(img_whitebg, img_garment_mask, model_prompt, bool_auto_mask, bool_face_detailer, prompt, lora, lora_strength):
    return cc.run_fashion_model_bg_at_once(
        img_whitebg=img_whitebg,
        img_garment_mask=img_garment_mask,
        model_prompt=model_prompt,
        bool_auto_mask=bool_auto_mask,
        bool_face_detailer=bool_face_detailer,
        prompt=prompt,
        lora=lora,
        lora_strength=lora_strength
    ).get('images')


def run_fashion_segmentation(img):
    return cc.run_fashion_segmentation(img=img).get('images')


def run_fashion_garment_detector(img):
    return cc.run_fashion_garment_detector(img=img)


def run_fashion_get_segments(img, bool_use_detection):
    return cc.run_fashion_get_segments(img=img, bool_use_detection=bool_use_detection)


def run_fashion_make_white_bg(image, mask):
    return cc.run_fashion_make_white_bg(image=image, mask=mask)


def run_fashion_preview_mask(img, mask):
    return cc.run_fashion_preview_mask(img=img, mask=mask)


def get_combined_mask(segmented_masks, indices, bool_use_auto_detected_mask=None, auto_detected_mask=None):
    masks_path_list = [path for path, _ in segmented_masks] if segmented_masks is not None else []
    return cc.get_combined_mask(segmented_masks=masks_path_list,
                                indices=indices,
                                bool_use_auto_detected_mask=bool_use_auto_detected_mask,
                                auto_detected_mask=auto_detected_mask)


def create_annotated_image(base_image, segments):
    annotations = []
    base_image = base_image.convert("RGBA")

    if segments is None:
        return base_image, annotations

    for idx, segment in enumerate(segments, start=1):
        segment_image = Image.open(segment[0]).convert("1")
        segment_np = np.array(segment_image)
        annotations.append((segment_np, str(idx)))

    return base_image, annotations


def on_annotation_click(clicked_indices_str, evt: gr.SelectData):
    index = str(evt.index + 1)
    clicked_indices_set = set(clicked_indices_str.split(","))
    clicked_indices_set.discard('')
    clicked_indices_set.add(index)
    result_clicked_indices_str = ",".join(clicked_indices_set)
    return result_clicked_indices_str


def toggle_interface(current_state):
    return not current_state


with gr.Blocks(analytics_enabled=False, title='imgthis') as demo5:
    with gr.Row():
        gr.Markdown("## AI Fashion Model Generation")

    # Image Input

    with gr.Row():
        img_fashion_input = gr.Image(type='pil', image_mode="RGBA")

    # Image Preprocess

    with gr.Group():
        with gr.Row():
            btn_fashion_bool_upscale = gr.Checkbox(label="Upscale", value=True)
            btn_fashion_preprocess = gr.Button("Preprocess Image", variant="primary")
        with gr.Row():
            img_fashion_upscale = gr.Image(type='pil', label='upscale', image_mode='RGBA')
            img_fashion_white_bg = gr.Image(type='pil', label='whiteBG', image_mode='RGBA')

    # RMBG Retrial

    fashion_state_show_rmbg_retrial = gr.State(False)
    btn_fashion_show_rmbg_retrial = gr.Button("Retry RMBG")

    with gr.Group(visible=False) as rmbg_retrial_interface:
        with gr.Row():
            btn_fashion_segmentation_rmbg_retry = gr.Button("Retry RMBG(wait after clicking)", variant="primary")
            fashion_segment_indices_rmbg_retry = gr.Textbox(label="Select Segments(Input numbers or Click the annotations)")
            btn_fashion_preview_rmbg_retry = gr.Button("View New RMBG", variant="primary")
        with gr.Row():
            imgs_fashion_segments_rmbg_retry = gr.Gallery(label='RMBG Segments', columns=[5], visible=False)
        with gr.Row():
            img_fashion_annotated_rmbg_retry = gr.AnnotatedImage(label="Annotated Image")
            img_fashion_combined_mask_rmbg_retry = gr.Image(type='pil', label='mask', image_mode='RGBA', visible=False)
            img_fashion_postprocessed_mask_rmbg_retry = gr.Image(type='pil', label='mask', image_mode='RGBA', visible=False)
            img_fashion_preview_rmbg_retry = gr.Image(type='pil', label='New RMBG', image_mode='RGBA')

    # Manual Masking

    fashion_state_show_manual_masking = gr.State(False)
    btn_fashion_show_manual_masking = gr.Button("Manual Masking")

    with gr.Group(visible=False) as manual_masking_interface:
        with gr.Row():
            fashion_use_detection = gr.Checkbox(label="Use Object Detection(takes more time)", value=False)
        with gr.Row():
            btn_fashion_segmentation = gr.Button("View Segments(wait after clicking)", variant="primary")
            fashion_segment_indices = gr.Textbox(label="Select Segments(Input numbers or Click the annotations)")
            btn_fashion_mask_preview = gr.Button("Confirm Mask", variant="primary")
        with gr.Row():
            imgs_fashion_segments = gr.Gallery(label='segment masks', columns=[5], visible=False)
            img_fashion_annotated_image_for_mask = gr.AnnotatedImage(label="Annotated Image")
            img_fashion_combined_mask = gr.Image(type='pil', label='combined mask', image_mode='RGBA', visible=False)
            img_fashion_garment_mask = gr.Image(type='pil', label='garment mask', image_mode='RGBA', visible=True)

    # Model and BG Generation

    with gr.Group():
        with gr.Row():
            fashion_face_prompt = gr.Textbox(label="Fashion Model Prompt")
            fashion_bg_prompt = gr.Textbox(label='Background prompt')
            fashion_bool_auto_mask = gr.Checkbox(label="Automatic Mask Generation", value=True)
            fashion_bool_detailer = gr.Checkbox(label="Add Face Details", value=False)
            fashion_lora = gr.Radio(['v1', 'v2', 'Not use'], value='v1', label="LoRA")
            fashion_lora_strength = gr.Slider(0, 1, value=0.90, step=0.01, label="LoRA Strength")

        with gr.Row():
            btn_fashion_bgic = gr.Button(f"Change Model and BG", variant="primary")

        with gr.Row():
            img_fashion_changbg_gallery = gr.Gallery(label="Generated Images", columns=[6], rows=[1], object_fit="contain", height="auto")

    # Control Buttons

    # Image Preprocess

    btn_fashion_preprocess.click(fn=run_fashion_gen_upscale, inputs=[img_fashion_input, btn_fashion_bool_upscale], outputs=img_fashion_upscale
                                 ).then(fn=run_fashion_make_white_bg_model, inputs=[img_fashion_upscale], outputs=img_fashion_white_bg)

    # RMBG Retrial

    btn_fashion_show_rmbg_retrial.click(toggle_interface, inputs=[fashion_state_show_rmbg_retrial], outputs=[fashion_state_show_rmbg_retrial], show_progress=False)

    fashion_state_show_rmbg_retrial.change(lambda visible: gr.update(visible=visible), inputs=[fashion_state_show_rmbg_retrial], outputs=rmbg_retrial_interface)

    btn_fashion_segmentation_rmbg_retry.click(fn=run_fashion_segmentation, inputs=[img_fashion_white_bg], outputs=imgs_fashion_segments_rmbg_retry
                                              ).then(fn=create_annotated_image, inputs=[img_fashion_white_bg, imgs_fashion_segments_rmbg_retry],
                                                     outputs=img_fashion_annotated_rmbg_retry)

    img_fashion_annotated_rmbg_retry.select(fn=on_annotation_click, inputs=[fashion_segment_indices_rmbg_retry], outputs=[fashion_segment_indices_rmbg_retry])

    btn_fashion_preview_rmbg_retry.click(fn=get_combined_mask, inputs=[imgs_fashion_segments_rmbg_retry, fashion_segment_indices_rmbg_retry], outputs=img_fashion_combined_mask_rmbg_retry
                                         ).then(fn=run_fashion_preview_mask, inputs=[img_fashion_white_bg, img_fashion_combined_mask_rmbg_retry], outputs=img_fashion_preview_rmbg_retry
                                                       ).then(fn=run_fashion_make_white_bg, inputs=[img_fashion_white_bg, img_fashion_combined_mask_rmbg_retry], outputs=[img_fashion_white_bg])

    # Manual Masking

    btn_fashion_show_manual_masking.click(toggle_interface, inputs=[fashion_state_show_manual_masking], outputs=[fashion_state_show_manual_masking], show_progress=False)

    fashion_state_show_manual_masking.change(lambda visible: gr.update(visible=visible), inputs=[fashion_state_show_manual_masking], outputs=manual_masking_interface)

    btn_fashion_segmentation.click(fn=run_fashion_get_segments, inputs=[img_fashion_white_bg, fashion_use_detection], outputs=imgs_fashion_segments
                                   ).then(fn=create_annotated_image, inputs=[img_fashion_white_bg, imgs_fashion_segments], outputs=img_fashion_annotated_image_for_mask)

    img_fashion_annotated_image_for_mask.select(fn=on_annotation_click, inputs=[fashion_segment_indices], outputs=[fashion_segment_indices])

    btn_fashion_mask_preview.click(fn=get_combined_mask, inputs=[imgs_fashion_segments, fashion_segment_indices], outputs=img_fashion_combined_mask
                                   ).then(fn=run_fashion_preview_mask, inputs=[img_fashion_white_bg, img_fashion_combined_mask], outputs=img_fashion_garment_mask)

    # Model and BG Generation

    btn_fashion_bgic.click(fn=run_fashion_change_model_bg_at_once, inputs=[img_fashion_white_bg, img_fashion_combined_mask, fashion_face_prompt, fashion_bool_auto_mask, 
                                                                           fashion_bool_detailer, fashion_bg_prompt, fashion_lora, fashion_lora_strength],
                                                                           outputs=img_fashion_changbg_gallery)

demo5.launch(server_name='0.0.0.0', server_port=54309)
