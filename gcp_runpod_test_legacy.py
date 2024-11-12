import json
import io
import time
import base64

import gradio as gr
import numpy as np
from PIL import Image

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

        elif server_type == 'runpod':
            self.base_url = 'https://api.runpod.ai/v2/'
            self.headers.update({'authorization': 'Bearer ' + RUNPOD_API_KEY})

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

    def submit_post_async(self, server, data, timeout=60):
        ret = {}
        # print(data)
        try:
            response = requests.post(f"{self.base_url}{RP_SERVER_ADDR.get(server)}/runsync",
                                     headers=self.headers,
                                     data=json.dumps(data).encode('utf-8'),
                                     timeout=(3, timeout))
            response.raise_for_status()

            response_body = response.json()

            rp_id = response_body.get('id')

            retries = 0
            while retries < timeout:
                rp_status = response_body.get('status')

                if rp_status in ['IN_PROGRESS', 'IN_QUEUE']:
                    time.sleep(1)
                    print(f"runpod status: {rp_status}")
                    response = requests.get(f"{self.base_url}{RP_SERVER_ADDR.get(server)}/status/{rp_id}",
                                            headers=self.headers,
                                            timeout=3
                                            )
                    response.raise_for_status()
                    response_body = response.json()
                    # print(response_body)

                elif rp_status == 'COMPLETED':
                    print(f"runpod's status: {rp_status}")
                    break
                else:
                    raise ValueError(f"Invalid Runpod Status: {rp_status}\n {response_body}")
                retries += 1

            comfyui_output_dict = response_body.get('output', {})

            if comfyui_output_dict.get('status') == 'error':
                print(f"An error occurred in ComfyUI: {comfyui_output_dict['message']}")
                for _details in comfyui_output_dict.get('details'):
                    print(_details, end='')

            elif comfyui_output_dict.get('status') == 'success':
                ret['outputs'] = comfyui_output_dict.get('outputs')

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
        image.save(buffered, format=ext)
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

    def run_gen_upscale(self, img):
        server = 'changebg' if self.server_type == 'runpod' else 'local'
        images = [img]
        data = self.fix_request_param('upscale', images)
        gen_dict = self.submit_post_sync(server, data)
        return {
            'image' : self.convert_base64url_to_img(gen_dict.get('outputs')[0].get('url'))
        }

    def run_gen_rmbg(self, img):
        server = 'changebg' if self.server_type == 'runpod' else 'local'
        images = [img]
        data = self.fix_request_param('rmbg', images)
        gen_dict = self.submit_post_sync(server, data)
        return {
            'image' : self.convert_base64url_to_img(gen_dict.get('outputs')[0].get('url'))
        }

    def run_gen_change_position(self, img):
        server = 'changebg' if self.server_type == 'runpod' else 'local'
        images = [img]
        data = self.fix_request_param('position', images)
        gen_dict = self.submit_post_sync(server, data)
        return {
            'image' : self.convert_base64url_to_img(gen_dict.get('outputs')[0].get('url'))
        }

    def run_gen_depth(self, img):
        server = 'changebg' if self.server_type == 'runpod' else 'local'
        images = [img]
        data = self.fix_request_param('depth', images)
        gen_dict = self.submit_post_sync(server, data)
        return {
            'image' : self.convert_base64url_to_img(gen_dict.get('outputs')[0].get('url'))
        }

    def run_gen_bgic(self, rmbg_img, depth_img, prompt, restoration_choices, lora, lora_strength, clip_skip):
        server = 'changebg' if self.server_type == 'runpod' else 'local'
        images = [rmbg_img, depth_img]
        lora_setting = {
            'lora' : lora,
            'strength' : lora_strength,
        }
        data = self.fix_request_param('bg', images, prompt=prompt, restoration_choices=restoration_choices, lora_setting=lora_setting, clip_skip=clip_skip)
        gen_dict = self.submit_post_sync(server, data)
        return {
            'images' : [(self.convert_base64url_to_img(image.get('url')), image.get('restoration_choice')) for image in gen_dict.get('outputs')]
        }

    def run_gen_vid(self, input_img, prompt, steps, fs, length):
        server = 'dyncraft' if self.server_type == 'runpod' else 'local'
        images = [input_img]
        frame = 16 if length == '2s' else 24
        i2v_setting = {
            'steps': steps,
            'fs': fs,
            'frames': frame,
            'frame_window_size': frame
        }
        data = self.fix_request_param('dynamic', images, promt=prompt, i2v_setting=i2v_setting)
        gen_dict = self.submit_post_async(server, data, timeout=180)
        return {
            'video': self.convert_base64url_to_vid(gen_dict.get('outputs')[0].get('url'))
        }

    def run_gen_vid_restore(self, input_img, input_vid, object_type):
        server = 'dyncraft' if self.server_type == 'runpod' else 'local'
        images = [input_img]
        video_urls = [self.convert_vid_to_base64url(input_vid)]
        data = self.fix_request_param('restore', images, object_type=object_type, video_urls=video_urls)
        gen_dict = self.submit_post_async(server, data)
        return {
            'video': self.convert_base64url_to_vid(gen_dict.get('outputs')[0].get('url'))
        }

    def run_m2m(self, img, mask, prompt):
        server = 'm2m' if self.server_type == 'runpod' else 'local'
        images = [img, mask]
        data = self.fix_request_param('m2m_api', images, prompt=prompt)
        gen_dict = self.submit_post_sync(server, data)
        return {
            'image' : self.convert_base64url_to_img(gen_dict.get('outputs')[0].get('url'))
        }

    def run_m2m_mask_bg(self, img):
        server = 'm2m' if self.server_type == 'runpod' else 'local'
        images = [img]
        data = self.fix_request_param('m2m_mask_bg', images)
        gen_dict = self.submit_post_sync(server, data)
        return {
            'image' : self.convert_base64url_to_img(gen_dict.get('outputs')[0].get('url'))
        }

    def run_m2m_at_once(self, masked_bg_img, prompt, restoration_choices, lora, lora_strength, clip_skip):
        server = 'm2m' if self.server_type == 'runpod' else 'local'
        images = [masked_bg_img]
        lora_setting = {
            'lora' : lora,
            'strength' : lora_strength,
        }
        data = self.fix_request_param('m2m_at_once', images, prompt=prompt, restoration_choices=restoration_choices, lora_setting=lora_setting, clip_skip=clip_skip)
        gen_dict = self.submit_post_sync(server, data)
        return {
            'images' : [(self.convert_base64url_to_img(image.get('url')), image.get('restoration_choice')) for image in gen_dict.get('outputs')]
        }

rc = RP_COMFY_Connector('local')
rc2 = RP_COMFY_Connector('local')
rc3 = RP_COMFY_Connector('local')

def run_gen_upscale(input_img):
    return rc.run_gen_upscale(input_img).get('image')

def run_gen_rmbg(input_img):
    return rc.run_gen_rmbg(input_img).get('image')

def run_gen_change_position(rmbg_img):
    return rc.run_gen_change_position(rmbg_img).get('image')

def run_gen_depth(rmbg_img):
    return rc.run_gen_depth(rmbg_img).get('image')

def run_gen_bgic(rmbg_img, depth_img, prompt, restoration_choices, lora, lora_strength, clip_skip):
    return rc.run_gen_bgic(rmbg_img, depth_img, prompt, restoration_choices, lora, lora_strength, clip_skip).get('images')

def run_gen_vid(input_img, prompt, steps, fs, length):
    return rc2.run_gen_vid(input_img, prompt, steps, fs, length).get('video')
 
def run_gen_vid_restore(input_img, input_vid, object_type):
    return rc2.run_gen_vid_restore(input_img, input_vid, object_type).get('video')

def run_m2m(input_img, prompt):

    background = input_img['background']
    layers = input_img['layers'][0]  

    mask = layers.split()[-1]  

    background = background.convert("RGBA")

    return rc3.run_m2m(mask, background, prompt).get('image')

def run_m2m_mask_bg(input_img):
    return rc3.run_m2m_mask_bg(input_img).get('image')

def run_m2m_at_once(masked_bg_img, prompt, restoration_choices, lora, lora_strength, clip_skip):
    return rc3.run_m2m_at_once(masked_bg_img, prompt, restoration_choices, lora, lora_strength, clip_skip).get('images')

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("## Image Generation")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type ='pil')
        with gr.Column(scale=3):
            with gr.Tab("Gen BG"):
                with gr.Row():
                    ic_upscale_img = gr.Image(type ='pil')
                    ic_rmbg_img = gr.Image(type ='pil', image_mode="RGBA")
                    ic_change_pos_img = gr.Image(type ='pil', image_mode="RGBA")
                    ic_depth_img = gr.Image(type ='pil')
                with gr.Row():
                    ic_text = gr.Textbox()
                    ic_restoration_choices = gr.CheckboxGroup(['Normal', 'Balance', 'Original'], label="Restoration")
                    ic_lora = gr.Radio(['v1', 'v2', 'Not use'], label="LoRA")
                    ic_lora_strength = gr.Slider(0, 1, value=0.90, step=0.01, label="LoRA Strength")
                    ic_clip_skip = gr.Checkbox(label="Clip Skip")
                    ic_btn = gr.Button(f"Generate BG")
                with gr.Row():
                    ic_changbg_gallery = gr.Gallery(
                        label="Generated Images",
                        columns=[3], rows=[1], object_fit="contain", height="auto"
                        )
                with gr.Row():
                    ic_regen_btn = gr.Button(f"ReGen Image")
                    ic_regen_btn.click(
                        run_gen_bgic,
                        inputs=[ic_change_pos_img, ic_depth_img, ic_text, ic_restoration_choices, ic_lora, ic_lora_strength, ic_clip_skip],
                        outputs=ic_changbg_gallery
                        )
                ic_btn.click(
                    run_gen_upscale,
                    inputs=[input_image], 
                    outputs=[ic_upscale_img]).then(
                        run_gen_rmbg,
                        inputs=[ic_upscale_img], 
                        outputs=[ic_rmbg_img]).then(
                            run_gen_change_position,
                            inputs=[ic_rmbg_img], 
                            outputs=[ic_change_pos_img]).then(
                                run_gen_depth,
                                inputs=[ic_change_pos_img],
                                outputs=[ic_depth_img]).then(
                                    run_gen_bgic,
                                    inputs=[ic_change_pos_img, ic_depth_img, ic_text, ic_restoration_choices, ic_lora, ic_lora_strength, ic_clip_skip],
                                    outputs=ic_changbg_gallery
                                    )
            with gr.Tab("Gen Video") as bg:
                with gr.Row():
                    gr.Markdown("## Video Generation")
                with gr.Row():
                    dy_upscale_img = gr.Image(type ='pil')
                    generate_upscale_button = gr.Button("Generate Upscale")
                    prompt = gr.Textbox(label = "prompt")
                with gr.Row():
                    steps = gr.Slider(15, 50, value = 30, step = 1, label="step")
                    fs = gr.Slider(2, 30, value = 7.5, step = 0.1, label="fs")
                    length = gr.Radio(['2s','3s'], value = '2s', label="video length")
                    generate_vid_button = gr.Button("Generate Video")
                with gr.Row():
                    with gr.Column():
                        video_output = gr.Video("Original Gen Video")
                        object_type = gr.Textbox(label = "Object_Type")
                        restore_vid_button = gr.Button("Detail Restore")
                    with gr.Column():
                        restore_video_output = gr.Video("Detail Restore Video")
                generate_upscale_button.click(
                    run_gen_upscale,
                    inputs=[input_image],
                    outputs=[dy_upscale_img]
                )
                generate_vid_button.click(
                    run_gen_vid, 
                    inputs=[dy_upscale_img,prompt,steps,fs,length], 
                    outputs=[video_output]
                    )
                restore_vid_button.click(
                    run_gen_vid_restore,
                    inputs=[dy_upscale_img,video_output,object_type],
                    outputs=[restore_video_output]
                )
            with gr.Tab("Mannequinne to Model") as bg:
                with gr.Row():
                    gr.Markdown("## Model Generation")
                with gr.Row():
                    m2m_input = gr.ImageMask(type = 'pil', image_mode="RGBA")
                with gr.Row():
                    m2m_text = gr.Textbox()
                    m2m_restoration_choices = gr.CheckboxGroup(['Normal', 'Balance', 'Original'], label="Restoration")
                    m2m_lora = gr.Radio(['v1', 'v2', 'Not use'], label="LoRA")
                    m2m_lora_strength = gr.Slider(0, 1, value=0.90, step=0.01, label="LoRA Strength")
                    m2m_clip_skip = gr.Checkbox(label="Clip Skip")
                with gr.Row():
                    prompt = gr.Textbox(label = "Prompt(Model)")
                    m2m_button = gr.Button("Generate Model")
                with gr.Row():
                    m2m_output = gr.Image(type = 'pil')
                    m2m_upscale_img = gr.Image(type = 'pil')
                    m2m_mask_bg = gr.Image(type = 'pil')
                with gr.Row():
                    m2m_changbg_gallery = gr.Gallery(
                        label="Generated Images",
                        columns=[3], rows=[1], object_fit="contain", height="auto"
                        )
                with gr.Row():
                    m2m_regen_btn = gr.Button(f"ReGen Image")
                    m2m_regen_btn.click(
                        run_m2m_at_once,
                        inputs=[m2m_mask_bg, m2m_text, m2m_restoration_choices, m2m_lora, m2m_lora_strength, m2m_clip_skip],
                        outputs=m2m_changbg_gallery
                        )
                m2m_button.click(
                    run_m2m,
                    inputs=[m2m_input, prompt],
                    outputs=[m2m_output]
                ).then(
                    run_gen_upscale,
                    inputs=[m2m_output], 
                    outputs=[m2m_upscale_img]).then(
                            run_m2m_mask_bg,
                            inputs=[m2m_upscale_img], 
                            outputs=[m2m_mask_bg]).then(
                                run_m2m_at_once,
                                inputs=[m2m_mask_bg, m2m_text, m2m_restoration_choices, m2m_lora, m2m_lora_strength, m2m_clip_skip],
                                outputs=m2m_changbg_gallery)
                        


demo.launch(server_name='0.0.0.0', server_port=54309)
