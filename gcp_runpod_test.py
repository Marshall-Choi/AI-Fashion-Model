import json
import io
import time
import base64
import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import requests

RUNPOD_API_KEY = 'MJKX9ULY5CGPIRZ2P4SY7CZI4PSOM5N6OJOYD54M'  # Pion AI

RP_SERVER_ADDR = {'m2m': '4dc8kcjhinteox',
                  'local' : '',
                  }


# Model


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

    def submit_post_sync(self, data, timeout=240):
        server=self.server
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

    @staticmethod
    def convert_img_to_base64url(image, ext='webp'):
        buffered = io.BytesIO()
        image.save(buffered, format=ext, lossless=True)
        return f'data:image/{ext};base64,' + base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    def convert_base64url_to_img(img_url):
        return Image.open(io.BytesIO(base64.b64decode(img_url.split(",")[1])))

    def fix_request_param(self, cmd_str, imgs, img_masks, **kwargs):
        image_urls = []
        for img in imgs:
            img_url = self.convert_img_to_base64url(img)
            image_urls.append(img_url)

        image_mask_urls = []
        for img_mask in img_masks:
            img_mask_url = self.convert_img_to_base64url(img_mask)
            image_mask_urls.append(img_mask_url)

        input_dict = {"command": cmd_str,
                      "image_urls": image_urls,
                      "image_mask_urls": image_mask_urls}
        input_dict.update(kwargs)

        return {"input": input_dict}

    def run_command(self, img_inputs, img_mask_inputs, **kwargs): 
        data = self.fix_request_param(cmd_str='change_model', imgs=img_inputs, img_masks=img_mask_inputs, **kwargs)
        gen_dict = self.submit_post_sync(data=data)
        return gen_dict
    
    # Generate Images

    def run_gen_upscale(self, img):
        gen_dict = self.run_command(img_inputs=[img], img_mask_inputs=[], workflow_select='upscale')
        return {
            'image' : self.convert_base64url_to_img(img_url=gen_dict.get('outputs')[0].get('url'))
        }

    def run_white_bg_model(self, img):
        gen_dict = self.run_command(img_inputs=[img], img_mask_inputs=[], workflow_select='whiteBG_model')
        return {
            'image' : self.convert_base64url_to_img(img_url=gen_dict.get('outputs')[0].get('url')),
            'mask' : self.convert_base64url_to_img(img_url=gen_dict.get('outputs')[1].get('url'))
        }

    def run_garment_detector(self, img):
        gen_dict = self.run_command(img_inputs=[img], img_mask_inputs=[], workflow_select='detection')
        return {
            'images' : [self.convert_base64url_to_img(image.get('url')) for image in gen_dict.get('outputs')]
        }

    def run_segmentation(self, img):
        gen_dict = self.run_command(img_inputs=[img], img_mask_inputs=[], workflow_select='segmentation')
        return {
            'images' : [self.convert_base64url_to_img(image.get('url')) for image in gen_dict.get('outputs')]
        }

    def run_change_model_bg(self, img_whitebg, img_rmbg_mask, img_selected_model, model_prompt, bg_prompt, bool_hand_detailer, img_garment_mask=None):
        img_inputs = [img_whitebg]
        img_mask_inputs = [img_rmbg_mask]

        if img_selected_model:
            img_inputs.append(img_selected_model)

        if img_garment_mask:
            img_mask_inputs.append(img_garment_mask)
        
        gen_dict = self.run_command(
            img_inputs=img_inputs, 
            img_mask_inputs=img_mask_inputs,
            workflow_select='genmodel', 
            model_prompt=model_prompt, 
            bg_prompt=bg_prompt,  
            use_hd=bool_hand_detailer
            )
        
        return {
            'image detail restored' : self.convert_base64url_to_img(gen_dict.get('outputs')[0].get('url')),
            'image clothing overlay' : self.convert_base64url_to_img(gen_dict.get('outputs')[1].get('url')),
            'mask' : self.convert_base64url_to_img(gen_dict.get('outputs')[2].get('url')),
            'pose skeleton' : self.convert_base64url_to_img(gen_dict.get('outputs')[3].get('url'))
        }


# Control


class Control:
    def __init__(self, server):
        self.rpcomfy = RP_COMFY_Connector(server)

    def run_fashion_gen_upscale(self, img):
        return self.rpcomfy.run_gen_upscale(img=img)

    def run_fashion_make_white_bg_model(self, img):
        return self.rpcomfy.run_white_bg_model(img=img)

    def run_fashion_segmentation(self, img):
        return self.rpcomfy.run_segmentation(img=img)

    def run_fashion_garment_detector(self, img):
        return self.rpcomfy.run_garment_detector(img=img)

    def run_fashion_get_segments(self, img, bool_use_detection):
        detected_masks = []
        if bool_use_detection:
            detected_masks = self.run_fashion_garment_detector(img=img).get('images')
        segmented_masks = self.run_fashion_segmentation(img=img).get('images')
        total_mask_list = detected_masks + segmented_masks
        return total_mask_list

    def get_combined_mask(self, paths_segmented_masks, segment_indices, image_for_size, img_drawn_mask_layer=None, bool_use_auto_detected_mask=None, auto_detected_mask=None):
        # Auto mask addition
        if bool_use_auto_detected_mask:
            base_mask = auto_detected_mask
        else:
            base_mask = Image.new("L", image_for_size.size, color=0)

        # Manual mask addition
        if img_drawn_mask_layer:
            mask_alpha_channel = img_drawn_mask_layer.split()[-1]
            base_mask = Image.composite(mask_alpha_channel, base_mask, mask_alpha_channel)

        # Segmented mask addition
        if not segment_indices:
            return base_mask
        
        segment_indices = [int(i) - 1 for i in segment_indices.split(',')]
        
        for index in segment_indices:
            segmented_mask_to_add = Image.open(paths_segmented_masks[index]).convert("L")
            base_mask = Image.composite(segmented_mask_to_add, base_mask, segmented_mask_to_add)

        return base_mask
    
    def clear_mask(self, img_for_size):
        return Image.new("L", img_for_size.size, color=0)

    def run_fashion_make_white_bg(self, img, mask_img):
        mask = mask_img.convert("L")
        inverted_mask = ImageOps.invert(mask)
        white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        masked_image = Image.composite(img, white_bg, inverted_mask)
        return masked_image

    def run_fashion_preview_mask(self, img, mask_image, mask_color=(100, 15, 240), mask_opacity=0.60):
        if mask_image is None:
            return img

        # Prepare images and mask
        img_rgba = img.convert("RGBA")  # Convert input image to RGBA
        mask = mask_image.convert('L')  # Convert mask to grayscale (L)
        mask_array = np.array(mask) / 255.0  # Normalize mask to 0~1
        mask_adjusted = mask_array * mask_opacity  # Adjust transparency

        # Create overlay and composite mask
        alpha_value = int(255 * mask_opacity)  # Calculate alpha for overlay
        overlay_image = Image.new("RGBA", img.size, mask_color + (alpha_value,))  # Create overlay with RGBA
        mask_composite = Image.fromarray((mask_adjusted * 255).astype(np.uint8), mode='L')  # Convert adjusted mask back to image

        # Composite the overlay with the input image
        preview_img = Image.composite(overlay_image, img_rgba, mask_composite)

        # Return final image in RGB format
        return preview_img.convert("RGB")

    def run_fashion_change_model_bg(self, img_whitebg, img_rmbg_mask, img_selected_model, model_prompt, bg_prompt, bool_hand_detailer, img_garment_mask=None):
        return self.rpcomfy.run_change_model_bg(
            img_whitebg=img_whitebg,
            img_rmbg_mask=img_rmbg_mask,
            img_selected_model=img_selected_model,
            model_prompt=model_prompt,
            bg_prompt=bg_prompt,
            bool_hand_detailer=bool_hand_detailer,
            img_garment_mask=img_garment_mask
        )


# View


cc = Control('local')


# Fashion  Begin


def run_fashion_gen_upscale(img, bool_upscale):
    return cc.run_fashion_gen_upscale(img=img).get('image') if bool_upscale else img


def run_fashion_make_white_bg_model(img):
    result = cc.run_fashion_make_white_bg_model(img=img)
    return result.get('image'), result.get('mask')


def run_fashion_change_model_bg(img_whitebg, img_rmbg_mask, img_selected_model, model_prompt, bg_prompt, bool_hand_detailer, img_garment_mask=None):
    result = cc.run_fashion_change_model_bg(
        img_whitebg=img_whitebg,
        img_rmbg_mask=img_rmbg_mask,
        img_selected_model=img_selected_model,
        model_prompt=model_prompt,
        bg_prompt=bg_prompt,
        bool_hand_detailer=bool_hand_detailer,
        img_garment_mask=img_garment_mask
    )
    return result.get('image detail restored'), result.get('image clothing overlay'), result.get('mask'), result.get('pose skeleton')


def run_fashion_segmentation(img):
    return cc.run_fashion_segmentation(img=img).get('images')


def run_fashion_get_segments(img, bool_use_detection):
    return cc.run_fashion_get_segments(img=img, bool_use_detection=bool_use_detection)


def run_fashion_make_white_bg(image, mask):
    return cc.run_fashion_make_white_bg(img=image, mask_img=mask)


def run_fashion_preview_mask(img, mask_image):
    return cc.run_fashion_preview_mask(img=img, mask_image=mask_image)


def get_combined_mask(segmented_masks, segment_indices, image_for_size, img_mask_preview=None, bool_use_auto_detected_mask=None, auto_detected_mask=None):
    list_masks_paths = [path for path, _ in segmented_masks] if segmented_masks else []
    list_mask_layers = img_mask_preview['layers'] if img_mask_preview else None
    img_drawn_mask_layer = list_mask_layers[0] if list_mask_layers else None
    return cc.get_combined_mask(paths_segmented_masks=list_masks_paths,
                                segment_indices=segment_indices,
                                image_for_size=image_for_size,
                                img_drawn_mask_layer=img_drawn_mask_layer,
                                bool_use_auto_detected_mask=bool_use_auto_detected_mask,
                                auto_detected_mask=auto_detected_mask)


def clear_mask(img_for_size):
    return cc.clear_mask(img_for_size=img_for_size)


def set_used_mask_as_default(img_auto_detected_mask):
    return img_auto_detected_mask


def create_annotated_image(base_image, segments):
    annotations = []
    base_image = base_image.convert("RGBA")

    for idx, segment in enumerate(segments, start=1):
        segment_image = Image.open(segment[0]).convert("1")
        segment_np = np.array(segment_image)
        annotations.append((segment_np, str(idx)))

    return base_image, annotations


def on_annotation_click(clicked_indices_str, evt: gr.SelectData):
    str_index = str(evt.index + 1)
    if clicked_indices_str:
        clicked_indices_str += ","
    clicked_indices_str += str_index
    return clicked_indices_str


def toggle_interface(current_state):
    return not current_state


def display_input_image(image):
    return image


def get_image_size(img):
    width, height = img.size
    return str(width), str(height)


def append_bg_prompt(btn_label):
    prompt_dict = {
        "Cobblestone street" : "A quaint cobblestone street lined with charming, old-world buildings painted in pastel colors. Flower boxes adorn the windows, and small cafés with outdoor seating spill onto the sidewalk.",
        "Lake" : "A cozy wooden cabin by a serene lake, surrounded by tall pines and soft morning light reflecting on still waters. A rustic dock and a wooden canoe complete the tranquil scene.",
        "Mountain peak" : "A towering mountain peak rises dramatically, its rugged rocky faces and scattered patches of snow catching the soft, golden glow of the first light of dawn.",
        "Forest" : "A serene forest shrouded in mist, with towering trees creating a lush green canopy. The forest floor is covered in soft moss and delicate wildflowers, offering a natural, tranquil atmosphere.",
        "Flower hill" : "A winding road meandering through rolling hills, adorned with golden fields and colorful wildflowers. The bright sun casts long, soft shadows, creating a warm, inviting glow over the landscape.",
        "Desert" : "A beautiful desert landscape with rolling sand dunes, their soft, golden sands gently shaped by the wind. In contrast to the vast desert, scattered pockets of green vegetation such as hardy cacti, small shrubs, and desert flowers bring life and color to the scene.",
        "Winter" : "A serene snowy winter landscape with tall, snow-covered pine trees, a frozen lake, and soft, powdery snow blanketing the ground. The crisp, fresh air adds to the tranquility of the scene, while the pale blue sky and gentle sunlight create a peaceful, inviting glow.",
        "Terrace" : "A beautiful terrace overlooking the sparkling blue sea, with terracotta pots scattered around, filled with lush plants. The scene exudes an air of tranquility and sophistication, with the sea breeze gently rustling the foliage.",
        "Garden" : "A vibrant, lush garden overflowing with greenery and vibrant flowers in full bloom. Stone pathways wind gracefully through dense foliage, lined with leafy plants, colorful blossoms, and fragrant shrubs that bring life and color to every corner.",
        "Urban" : "A bustling urban street lined with sleek, high-rise buildings and trendy cafes. Neon signs flicker in the background, casting a colorful glow.",
        "Cafe" : "A cozy, stylish café with warm lighting and wooden furniture. The walls are adorned with modern art pieces, and there are plants and greenery.",
        "Luxury garden" : "A lavish garden adorned with manicured hedges, blooming rose bushes, and a grand marble fountain at its center. Elegant stone pathways lead to a luxurious mansion in the background, its facade adorned with tall, arched windows and intricate details."
        }

    return prompt_dict.get(btn_label)


def on_image_select_model(evt: gr.SelectData):
    model_img_path = evt._data['value']['image']['path']
    
    model_prompts = [
        "North American female model with pale skin, blue eyes and long waving blonde hair and no facial expression",
        "An Asian female fashion model with waving black hair styled in a sleek bob and slender face",
        "An African female fashion model wearing a soft, confident smile ",
        "a north American pale-skinned female fashion model with dark red hair, styled in loose waves",
        "A Japanese model with short bob hair with bangs, a graceful expression and minimal makeup",
        "caucasian, A masculine white north American male model with black short quiff hair, white skin",
        "A lean white male model with light blonde hair and piercing blue eyes. His hair is styled in a neat medium-length wave, complementing his clean-shaven face",
        "A striking black male model with rich ebony skin and short, tightly coiled hair. His strong jawline is accentuated by a well-groomed beard",
        "A Korean male model with smooth, fair skin and monolid eyes. His straight black hair is neatly styled in a classic side part, complementing his sharp cheekbones and clean-shaven face",
        "A Japanese male model with sharp, defined features and jet-black hair styled into a messy fringe. His hairstyle perfectly complements his clean-shaven face and slender build"
    ]
    
    model_prompt = model_prompts[evt.index]
    return model_img_path, model_prompt


with gr.Blocks(analytics_enabled=False, title='imgthis') as demo5:
    with gr.Row():
        gr.Markdown("## AI Fashion Model Generation")

    # Settings

    with gr.Tabs():
        with gr.Tab("Fashion Model Prompt"):
            with gr.Row():
                image_urls = [
                                "https://raw.githubusercontent.com/pioncmr/image-repo/main/female_model01.png",
                                "https://raw.githubusercontent.com/pioncmr/image-repo/main/female_model02.png",
                                "https://raw.githubusercontent.com/pioncmr/image-repo/main/female_model03.png",
                                "https://raw.githubusercontent.com/pioncmr/image-repo/main/female_model04.png",
                                "https://raw.githubusercontent.com/pioncmr/image-repo/main/female_model05.png",
                                "https://raw.githubusercontent.com/pioncmr/image-repo/main/male_model01.png",
                                "https://raw.githubusercontent.com/pioncmr/image-repo/main/male_model02.png",
                                "https://raw.githubusercontent.com/pioncmr/image-repo/main/male_model03.png",
                                "https://raw.githubusercontent.com/pioncmr/image-repo/main/male_model04.png",
                                "https://raw.githubusercontent.com/pioncmr/image-repo/main/male_model05.png"
                            ]
                imgs_fashion_models = gr.Gallery(
                    label="Select model", 
                    columns=5,
                    value=image_urls)

                img_fashion_selected_model = gr.Image(label="Selected model", type="pil")

        with gr.Tab("Background Prompt"):
            with gr.Row():
                btn_bg_prompt_1 = gr.Button("Cobblestone street")
                btn_bg_prompt_2 = gr.Button("Lake")
                btn_bg_prompt_3 = gr.Button("Mountain peak")
                btn_bg_prompt_4 = gr.Button("Forest")
                btn_bg_prompt_5 = gr.Button("Flower hill")
                btn_bg_prompt_6 = gr.Button("Desert")
                btn_bg_prompt_7 = gr.Button("Winter")
                btn_bg_prompt_8 = gr.Button("Terrace")
                btn_bg_prompt_9 = gr.Button("Garden")
                btn_bg_prompt_10 = gr.Button("Urban")
                btn_bg_prompt_11 = gr.Button("Cafe")
                btn_bg_prompt_12 = gr.Button("Luxury garden")

    with gr.Row():
        fashion_model_prompt = gr.Textbox(label="Fashion Model Prompt", value="A fashion model")
        fashion_bg_prompt = gr.Textbox(label="Background Prompt", value="A quaint cobblestone street lined with charming, old-world buildings painted in pastel colors. Flower boxes adorn the windows, and small cafés with outdoor seating spill onto the sidewalk.")
        btn_fashion_bool_hand_detailer = gr.Checkbox(label="Use Hand Refiner", value=False)

    # Image Input

    with gr.Group():
        with gr.Row():
            img_fashion_input = gr.Image(type='pil', image_mode="RGBA")
        with gr.Row():
            img_fashion_input_width = gr.Textbox(label="Width", value=str(img_fashion_input.width))
            img_fashion_input_height = gr.Textbox(label="Height", value=str(img_fashion_input.height))

    # Image Preprocess

    with gr.Group():
        with gr.Row():
            btn_fashion_bool_upscale = gr.Checkbox(label="Upscale", value=False)
            btn_fashion_preprocess = gr.Button("Preprocess Image", variant="primary")
        with gr.Row():
            img_fashion_upscale = gr.Image(type='pil', label='Upscale', image_mode='RGBA')
            img_fashion_white_bg = gr.Image(type='pil', label='WhiteBG', image_mode='RGBA')
            img_fashion_rmbg_mask = gr.Image(type='pil', label='RMBG mask', image_mode='RGBA', visible=False)
        with gr.Row():
            img_fashion_upscale_width = gr.Textbox(label="Width")
            img_fashion_upscale_height = gr.Textbox(label="Height")

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
            img_fashion_annotated_rmbg_retry = gr.AnnotatedImage(label="Annotated Image", height="auto")
            img_fashion_combined_mask_rmbg_retry = gr.Image(type='pil', label='mask', image_mode='RGBA', visible=False, height="auto")
            img_fashion_postprocessed_mask_rmbg_retry = gr.Image(type='pil', label='mask', image_mode='RGBA', visible=False, height="auto")
            img_fashion_preview_rmbg_retry = gr.Image(type='pil', label='New RMBG', image_mode='RGBA', height="auto")

    # Manual Masking

    fashion_state_show_manual_masking = gr.State(False)
    btn_fashion_show_manual_masking = gr.Button("Manual Masking")

    with gr.Group(visible=False) as manual_masking_interface:
        with gr.Row():
            fashion_use_detection = gr.Checkbox(label="Use Object Detection(takes more time)", value=False)
            fashion_use_auto_detected_mask = gr.Checkbox(label="Use Auto Detected Mask", value=True)
        with gr.Row():
            btn_fashion_segmentation = gr.Button("View Segments(wait after clicking)", variant="primary")
            fashion_segment_indices = gr.Textbox(label="Select Segments(Input numbers or Click the annotations)")
            btn_fashion_mask_preview = gr.Button("Confirm Mask", variant="primary")
            btn_fashion_mask_clear = gr.Button("Clear Mask", variant="primary")
        with gr.Row():
            imgs_fashion_segments = gr.Gallery(label='segment masks', columns=[5], visible=False, height="auto")
            img_fashion_pose_skeleton = gr.Image(type='pil', label='Skeleton', image_mode='RGBA', height="auto")
            img_fashion_annotated_image_for_mask = gr.AnnotatedImage(label="Annotated Image", height="auto")
            img_fashion_combined_mask = gr.Image(type='pil', label='combined mask', image_mode='RGBA', visible=False, height="auto")
            img_fashion_garment_mask_preview = gr.ImageMask(type='pil', label='garment mask', image_mode='RGBA', visible=True, interactive=True, height="auto")

    # Model and BG Generation

    with gr.Group():
        with gr.Row():
            btn_fashion_change_model = gr.Button(f"Change Model and BG", variant="primary")

        with gr.Row():
            img_fashion_input_display = gr.Image(type='pil', label='Input image', image_mode='RGBA', height="auto")
            img_fashion_changed_background = gr.Image(type='pil', label="Detail restored", image_mode='RGBA', height='auto')
            img_fashion_clothing_overlay = gr.Image(type='pil', label="Clothing overlay", image_mode='RGBA', height='auto')
            img_fashion_auto_detected_mask = gr.Image(type='pil', label='Used mask', image_mode='RGBA', visible=False, height="auto")

    # Control Buttons

    img_fashion_input.upload(fn=get_image_size, inputs=img_fashion_input, outputs=[img_fashion_input_width, img_fashion_input_height]
                             ).then(fn=run_fashion_gen_upscale, inputs=[img_fashion_input, btn_fashion_bool_upscale], outputs=img_fashion_upscale
                                    ).then(fn=run_fashion_make_white_bg_model, inputs=[img_fashion_upscale], outputs=[img_fashion_white_bg, img_fashion_rmbg_mask]
                                           ).then(fn=get_image_size, inputs=img_fashion_upscale, outputs=[img_fashion_upscale_width, img_fashion_upscale_height]
                                                  ).then(fn=clear_mask, inputs=[img_fashion_white_bg], outputs=[img_fashion_combined_mask]
                                                         ).then(fn=run_fashion_change_model_bg, 
                                                                inputs=[img_fashion_white_bg, 
                                                                img_fashion_rmbg_mask, 
                                                                img_fashion_selected_model, 
                                                                fashion_model_prompt, 
                                                                fashion_bg_prompt, 
                                                                btn_fashion_bool_hand_detailer],
                                                                        outputs=[img_fashion_changed_background, img_fashion_clothing_overlay, img_fashion_auto_detected_mask, img_fashion_pose_skeleton]
                                                                        ).then(fn=set_used_mask_as_default, inputs=img_fashion_auto_detected_mask, outputs=img_fashion_combined_mask
                                                                        ).then(display_input_image, inputs=[img_fashion_input], outputs=[img_fashion_input_display])

    # Image Preprocess

    btn_fashion_preprocess.click(fn=run_fashion_gen_upscale, inputs=[img_fashion_input, btn_fashion_bool_upscale], outputs=img_fashion_upscale
                                 ).then(fn=run_fashion_make_white_bg_model, inputs=[img_fashion_upscale], outputs=[img_fashion_white_bg, img_fashion_rmbg_mask]
                                        ).then(fn=get_image_size, inputs=img_fashion_upscale, outputs=[img_fashion_upscale_width, img_fashion_upscale_height])

    # Model Selection

    imgs_fashion_models.select(on_image_select_model, inputs=[], outputs=[img_fashion_selected_model, fashion_model_prompt])

    # BG Prompt Selection

    btn_bg_prompt_1.click(fn=append_bg_prompt, inputs=[btn_bg_prompt_1], outputs=fashion_bg_prompt)
    btn_bg_prompt_2.click(fn=append_bg_prompt, inputs=[btn_bg_prompt_2], outputs=fashion_bg_prompt)
    btn_bg_prompt_3.click(fn=append_bg_prompt, inputs=[btn_bg_prompt_3], outputs=fashion_bg_prompt)
    btn_bg_prompt_4.click(fn=append_bg_prompt, inputs=[btn_bg_prompt_4], outputs=fashion_bg_prompt)
    btn_bg_prompt_5.click(fn=append_bg_prompt, inputs=[btn_bg_prompt_5], outputs=fashion_bg_prompt)
    btn_bg_prompt_6.click(fn=append_bg_prompt, inputs=[btn_bg_prompt_6], outputs=fashion_bg_prompt)
    btn_bg_prompt_7.click(fn=append_bg_prompt, inputs=[btn_bg_prompt_7], outputs=fashion_bg_prompt)
    btn_bg_prompt_8.click(fn=append_bg_prompt, inputs=[btn_bg_prompt_8], outputs=fashion_bg_prompt)
    btn_bg_prompt_9.click(fn=append_bg_prompt, inputs=[btn_bg_prompt_9], outputs=fashion_bg_prompt)
    btn_bg_prompt_10.click(fn=append_bg_prompt, inputs=[btn_bg_prompt_10], outputs=fashion_bg_prompt)
    btn_bg_prompt_11.click(fn=append_bg_prompt, inputs=[btn_bg_prompt_11], outputs=fashion_bg_prompt)
    btn_bg_prompt_12.click(fn=append_bg_prompt, inputs=[btn_bg_prompt_12], outputs=fashion_bg_prompt)
    
    # RMBG Retrial

    btn_fashion_show_rmbg_retrial.click(toggle_interface, inputs=[fashion_state_show_rmbg_retrial], outputs=[fashion_state_show_rmbg_retrial], show_progress=False)

    fashion_state_show_rmbg_retrial.change(lambda visible: gr.update(visible=visible), inputs=[fashion_state_show_rmbg_retrial], outputs=rmbg_retrial_interface)

    btn_fashion_segmentation_rmbg_retry.click(fn=run_fashion_segmentation, inputs=[img_fashion_white_bg], outputs=imgs_fashion_segments_rmbg_retry
                                              ).then(fn=create_annotated_image, inputs=[img_fashion_white_bg, imgs_fashion_segments_rmbg_retry],
                                                     outputs=img_fashion_annotated_rmbg_retry)

    img_fashion_annotated_rmbg_retry.select(fn=on_annotation_click, inputs=[fashion_segment_indices_rmbg_retry], outputs=[fashion_segment_indices_rmbg_retry])

    btn_fashion_preview_rmbg_retry.click(fn=get_combined_mask, inputs=[imgs_fashion_segments_rmbg_retry, fashion_segment_indices_rmbg_retry, img_fashion_white_bg], outputs=img_fashion_combined_mask_rmbg_retry
                                         ).then(fn=run_fashion_preview_mask, inputs=[img_fashion_white_bg, img_fashion_combined_mask_rmbg_retry], outputs=img_fashion_preview_rmbg_retry
                                                       ).then(fn=run_fashion_make_white_bg, inputs=[img_fashion_white_bg, img_fashion_combined_mask_rmbg_retry], outputs=[img_fashion_white_bg])

    # Manual Masking

    btn_fashion_show_manual_masking.click(toggle_interface, inputs=[fashion_state_show_manual_masking], outputs=[fashion_state_show_manual_masking], show_progress=False)

    fashion_state_show_manual_masking.change(lambda visible: gr.update(visible=visible), inputs=[fashion_state_show_manual_masking], outputs=manual_masking_interface)

    btn_fashion_segmentation.click(fn=run_fashion_get_segments, inputs=[img_fashion_white_bg, fashion_use_detection], outputs=imgs_fashion_segments
                                   ).then(fn=create_annotated_image, inputs=[img_fashion_white_bg, imgs_fashion_segments], outputs=img_fashion_annotated_image_for_mask)

    img_fashion_annotated_image_for_mask.select(fn=on_annotation_click, inputs=[fashion_segment_indices], outputs=[fashion_segment_indices])

    btn_fashion_mask_preview.click(fn=get_combined_mask, inputs=[imgs_fashion_segments, fashion_segment_indices, img_fashion_white_bg, img_fashion_garment_mask_preview, fashion_use_auto_detected_mask, img_fashion_auto_detected_mask], outputs=img_fashion_combined_mask
                                   ).then(fn=run_fashion_preview_mask, inputs=[img_fashion_white_bg, img_fashion_combined_mask], outputs=img_fashion_garment_mask_preview)

    btn_fashion_mask_clear.click(fn=clear_mask, inputs=[img_fashion_white_bg], outputs=[img_fashion_combined_mask]
                                   ).then(fn=run_fashion_preview_mask, inputs=[img_fashion_white_bg, img_fashion_combined_mask], outputs=img_fashion_garment_mask_preview)

    # Model and BG Generation

    btn_fashion_change_model.click(fn=run_fashion_change_model_bg, 
                           inputs=[img_fashion_white_bg, 
                           img_fashion_rmbg_mask, 
                           img_fashion_selected_model, 
                           fashion_model_prompt, 
                           fashion_bg_prompt, 
                           btn_fashion_bool_hand_detailer, 
                           img_fashion_combined_mask], 
                           outputs=[img_fashion_changed_background, 
                           img_fashion_clothing_overlay, 
                           img_fashion_auto_detected_mask, 
                           img_fashion_pose_skeleton]
                           ).then(fn=set_used_mask_as_default, inputs=img_fashion_auto_detected_mask, outputs=img_fashion_combined_mask)

demo5.launch(server_name='0.0.0.0', server_port=54309)
