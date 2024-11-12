    def run_command(self, cmd_str, inputs, **kwargs):
        data = self.fix_request_param(cmd_str=cmd_str, images=inputs, **kwargs)
        gen_dict = self.submit_post_sync(server='m2m', data=data)
        return gen_dict

    def run_garment_detector(self, img, image_selection):
        gen_dict = self.run_command(cmd_str='GenModel_detection', inputs=[img], prompt=image_selection)
        if image_selection == "":
            return {
                'image': RP_COMFY_Connector.convert_base64url_to_img(img_url=gen_dict.get('outputs')[0].get('url')),
                'index': gen_dict.get('aux')
            }
        else:
            return {
                'image': RP_COMFY_Connector.convert_base64url_to_img(img_url=gen_dict.get('outputs')[0].get('url'))
            }

    def run_whiteBG_garment(self, img, mask):
        gen_dict = self.run_command(cmd_str='GenModel_whiteBG_garment', inputs=[img, mask])
        return {
            'image': RP_COMFY_Connector.convert_base64url_to_img(img_url=gen_dict.get('outputs')[0].get('url'))
        }

    def run_gen_model(self, img_fashion_upscale, img_fashion_whiteBG_garment, garment_mask, prompt):
        gen_dict = self.run_command(cmd_str='GenModel_genmodel', inputs=[img_fashion_upscale, img_fashion_whiteBG_garment, garment_mask], prompt=prompt)
        return {
            'image': RP_COMFY_Connector.convert_base64url_to_img(img_url=gen_dict.get('outputs')[0].get('url'))
        }

    def run_white_bg_model(self, model_img):
        gen_dict = self.run_command(cmd_str='GenModel_whiteBG_model', inputs=[model_img])
        return {
            'image': RP_COMFY_Connector.convert_base64url_to_img(img_url=gen_dict.get('outputs')[0].get('url'))
        }

    def run_face_detailer(self, img_fashion_whiteBG_model, img_fashion_whiteBG_garment, garment_mask, prompt):
        gen_dict = self.run_command(cmd_str='GenModel_detailer', inputs=[img_fashion_whiteBG_model, img_fashion_whiteBG_garment, garment_mask], prompt=prompt)
        return {
            'image': RP_COMFY_Connector.convert_base64url_to_img(img_url=gen_dict.get('outputs')[0].get('url'))
        }

    def run_gen_fashion_bg(self, masked_bg_img, img_fashion_whiteBG_garment, img_fashion_garment_mask, prompt, restoration_choices, lora, lora_strength, clip_skip):
        lora_setting = {
            'lora': lora,
            'strength': lora_strength,
        }
        gen_dict = self.run_command(cmd_str='GenModel_genBG', inputs=[masked_bg_img, img_fashion_whiteBG_garment, img_fashion_garment_mask],
                                    prompt=prompt, restoration_choices=restoration_choices, lora_setting=lora_setting, clip_skip=clip_skip)
        return {
            'images': [(RP_COMFY_Connector.convert_base64url_to_img(image.get('url')), image.get('restoration_choice')) for image in gen_dict.get('outputs')]
        }

    def run_fashion_segmentation(self, img):
        gen_dict = self.run_command(cmd_str='GenModel_segmentation', inputs=[img])
        return {
            'images': [RP_COMFY_Connector.convert_base64url_to_img(image.get('url')) for image in gen_dict.get('outputs')]
        }

    def run_preview_mask(self, img, mask):
        gen_dict = self.run_command(cmd_str='GenModel_preview_mask', inputs=[img, mask])
        return {
            'image': RP_COMFY_Connector.convert_base64url_to_img(img_url=gen_dict.get('outputs')[0].get('url'))
        }

    def run_postprocess_mask(self, mask):
        gen_dict = self.run_command(cmd_str='GenModel_grow_mask', inputs=[mask])
        return {
            'image': RP_COMFY_Connector.convert_base64url_to_img(img_url=gen_dict.get('outputs')[0].get('url'))
        }

    def run_auto_mask(self, img):
        gen_dict = self.run_command(cmd_str='GenModel_auto_mask', inputs=[img])
        return {
            'image': RP_COMFY_Connector.convert_base64url_to_img(img_url=gen_dict.get('outputs')[0].get('url'))
        }



################

# Control

class Control:

    def run_fashion_make_white_bg(self, img):
        return self.rpcomfy.run_white_bg_model(img)

    def run_whiteBG_garment(self, img, mask):
        return self.rpcomfy.run_whiteBG_garment(img=img, mask=mask)

    def run_gen_model(self, img_fashion_upscale, img_fashion_whiteBG_garment, garment_mask, prompt):
        return self.rpcomfy.run_gen_model(img_fashion_upscale=img_fashion_upscale,
                                          img_fashion_whiteBG_garment=img_fashion_whiteBG_garment,
                                          garment_mask=garment_mask,
                                          prompt=prompt)

    def run_face_detailer(self, img_fashion_whiteBG_model, img_fashion_whiteBG_garment, garment_mask, prompt):
        return self.rpcomfy.run_face_detailer(img_fashion_whiteBG_model=img_fashion_whiteBG_model,
                                              img_fashion_whiteBG_garment=img_fashion_whiteBG_garment,
                                              garment_mask=garment_mask,
                                              prompt=prompt)

    def run_fashion_gen_BG(self, masked_bg_img, img_fashion_whiteBG_garment, img_fashion_garment_mask, prompt, restoration_choices, lora, lora_strength, clip_skip):
        return self.rpcomfy.run_gen_fashion_bg(masked_bg_img=masked_bg_img,
                                               img_fashion_whiteBG_garment=img_fashion_whiteBG_garment,
                                               img_fashion_garment_mask=img_fashion_garment_mask,
                                               prompt=prompt,
                                               restoration_choices=restoration_choices,
                                               lora=lora,
                                               lora_strength=lora_strength,
                                               clip_skip=clip_skip)
        # return self.rpcomfy.run_gen_fashion_bg(masked_bg_img, prompt, restoration_choices, lora, lora_strength, clip_skip)

    def run_fashion_segmentation(self, img):
        return self.rpcomfy.run_fashion_segmentation(img=img)

    def run_preview_mask(self, img, mask):
        return self.rpcomfy.run_preview_mask(img=img, mask=mask)

    def run_fashion_postprocess_mask(self, mask):
        return self.rpcomfy.run_postprocess_mask(mask=mask)

    def run_fashion_garment_detector(self, img):
        obj_list = []
        num_objects = self.rpcomfy.run_garment_detector(img=img, image_selection="").get('index')
        for i in range(num_objects):
            detected_obj = self.rpcomfy.run_garment_detector(img=img, image_selection=str(i)).get('image')
            obj_list.append(detected_obj)
        return obj_list

    def run_fashion_get_segments(self, img):
        detected_masks = self.run_fashion_garment_detector(img=img)
        segmented_masks = self.run_fashion_segmentation(img=img)
        total_mask_list = detected_masks + segmented_masks
        return total_mask_list

    def run_auto_mask(self, img):
        return self.rpcomfy.run_auto_mask(img=img)



###################
# View


def run_fashion_gen_upscale(img, bool_upscale):
    return cc.run_gen_upscale(img=img).get('image') if bool_upscale else img


def run_fashion_make_white_bg(img):
    return cc.run_fashion_make_white_bg(img=img).get('image')


def run_fashion_whiteBG_garment(img, mask):
    return cc.run_whiteBG_garment(img=img, mask=mask).get('image')


def run_fashion_gen_model(img_fashion_upscale, img_fashion_whiteBG_garment, garment_mask, face_prompt):
    return cc.run_gen_model(img_fashion_upscale=img_fashion_upscale,
                            img_fashion_whiteBG_garment=img_fashion_whiteBG_garment,
                            garment_mask=garment_mask,
                            prompt=face_prompt).get('image')


def run_fashion_face_detailer(img_fashion_whiteBG_model, img_fashion_whiteBG_garment, garment_mask, prompt):
    return cc.run_face_detailer(img_fashion_whiteBG_model=img_fashion_whiteBG_model,
                                img_fashion_whiteBG_garment=img_fashion_whiteBG_garment,
                                garment_mask=garment_mask,
                                prompt=prompt).get('image')


def run_fashion_change_BG(masked_bg_img, img_fashion_whiteBG_garment, img_fashion_garment_mask, prompt, restoration_choices, lora, lora_strength, clip_skip):
    return cc.run_fashion_gen_BG(masked_bg_img=masked_bg_img, img_fashion_whiteBG_garment=img_fashion_whiteBG_garment, img_fashion_garment_mask=img_fashion_garment_mask, prompt=prompt, restoration_choices=restoration_choices, lora=lora, lora_strength=lora_strength, clip_skip=clip_skip).get('images')


def run_fashion_segmentation(img):
    return cc.run_fashion_segmentation(img=img).get('images')


def run_fashion_preview_mask(img, mask):
    return cc.run_preview_mask(img=img, mask=mask).get('image')


def run_fashion_postprocess_mask(mask):
    return cc.run_fashion_postprocess_mask(mask=mask).get('image')


def run_fashion_get_segments(img):
    return cc.run_fashion_get_segments(img=img)


def run_fashion_auto_mask(img):
    return cc.run_auto_mask(img=img).get('image')


def get_combined_mask(segmented_masks, indices, skip_auto_mask, auto_mask, rmbg_retrial):
    def load_mask(mask_path):
        return Image.open(mask_path).convert("L")

    if skip_auto_mask or rmbg_retrial == gr.State(True):
        auto_mask = Image.new("L", load_mask(segmented_masks[0][0]).size, color=0)

    if not indices:
        return auto_mask

    indices = [int(i) - 1 for i in indices.split(',')]

    for index in indices:
        mask_image = load_mask(segmented_masks[index][0])
        auto_mask = Image.composite(mask_image, auto_mask, mask_image)

    return auto_mask


def create_annotated_image(base_image, segments, clicked_indices):
    clicked_indices.clear()
    annotations = []
    base_image = base_image.convert("RGBA")

    if segments is None:
        return [base_image, annotations, clicked_indices]

    for idx, segment in enumerate(segments, start=1):
        segment_image = Image.open(segment[0]).convert("1")
        # binary_segment = np.where(np.array(segment_image) > 128, 1, 0)
        annotations.append((segment_image, str(idx)))

    return [base_image, annotations, clicked_indices]


def on_annotation_click(clicked_indices, evt: gr.SelectData):
    index = evt.index + 1

    if index not in clicked_indices:
        clicked_indices.append(index)

    return [','.join(map(str, clicked_indices)), clicked_indices]


def toggle_interface(current_state):
    return not current_state


with gr.Blocks(analytics_enabled=False, title='imgthis', theme='Medguy/base2') as demo5:
    with gr.Row():
        gr.Markdown("## AI Fashion Model Generation")

    # Image Input

    with gr.Row():
        img_fashion_input = gr.Image(type='pil', image_mode="RGB")

    # Image Preprocess

    with gr.Group():
        with gr.Row():
            btn_fashion_bool_upscale = gr.Checkbox(label="Upscale", value=False)
            btn_fashion_preprocess = gr.Button("Preprocess Image", variant="primary")
        with gr.Row():
            img_fashion_upscale = gr.Image(type='pil', label='upscale', image_mode='RGB')
            img_fashion_white_bg = gr.Image(type='pil', label='whiteBG', image_mode='RGB')

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
            fashion_skip_auto_mask = gr.Checkbox(label="Not Use Auto Mask", value=False)
        with gr.Row():
            btn_fashion_segmentation = gr.Button("View Segments(wait after clicking)", variant="primary")
            fashion_segment_indices = gr.Textbox(label="Select Segments(Input numbers or Click the annotations)")
            btn_fashion_mask_preview = gr.Button("View Mask", variant="primary")
        with gr.Row():
            img_fashion_segments = gr.Gallery(label='segment masks', columns=[5], visible=True)
        with gr.Row():
            img_fashion_annotated_image_for_mask = gr.AnnotatedImage(label="Annotated Image")
            img_fashion_combined_mask = gr.Image(type='pil', label='combined mask', image_mode='RGBA', visible=True)
            img_fashion_combined_mask_preview = gr.Image(type='pil', label='combined mask preview', image_mode='RGBA')
            img_fashion_garment_mask = gr.Image(type='pil', label='garment mask', image_mode='RGBA', visible=True)
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

    btn_fashion_preprocess.click(fn=run_fashion_gen_upscale, inputs=[img_fashion_input, btn_fashion_bool_upscale], outputs=img_fashion_upscale
                                 ).then(fn=run_fashion_make_white_bg, inputs=img_fashion_upscale, outputs=img_fashion_white_bg)

    # RMBG Retrial

    btn_fashion_show_rmbg_retrial.click(toggle_interface, inputs=[fashion_state_show_rmbg_retrial], outputs=[fashion_state_show_rmbg_retrial], show_progress=False)

    fashion_state_show_rmbg_retrial.change(lambda visible: gr.update(visible=visible), inputs=[fashion_state_show_rmbg_retrial], outputs=rmbg_retrial_interface)

    clicked_indices = gr.State([])
    btn_fashion_segmentation_rmbg_retry.click(fn=run_fashion_segmentation, inputs=[img_fashion_white_bg], outputs=img_fashion_segments_rmbg_retry
                                              ).then(fn=create_annotated_image, inputs=[img_fashion_white_bg, img_fashion_segments_rmbg_retry, clicked_indices],
                                                     outputs=[img_fashion_annotated_rmbg_retry, clicked_indices])

    img_fashion_annotated_rmbg_retry.select(fn=on_annotation_click, inputs=clicked_indices, outputs=[fashion_segment_indices_rmbg_retry, clicked_indices])

    btn_fashion_preview_rmbg_retry.click(fn=get_combined_mask, inputs=[img_fashion_segments_rmbg_retry, fashion_segment_indices_rmbg_retry, fashion_skip_auto_mask, img_fashion_auto_mask, fashion_state_show_rmbg_retrial], outputs=img_fashion_combined_mask_rmbg_retry
                                         ).then(fn=run_fashion_preview_mask, inputs=[img_fashion_white_bg, img_fashion_combined_mask_rmbg_retry], outputs=img_fashion_preview_rmbg_retry
                                                ).then(fn=run_fashion_postprocess_mask, inputs=img_fashion_combined_mask_rmbg_retry, outputs=img_fashion_postprocessed_mask_rmbg_retry
                                                       ).then(fn=run_fashion_whiteBG_garment, inputs=[img_fashion_white_bg, img_fashion_combined_mask_rmbg_retry], outputs=[img_fashion_white_bg])

    # Image Auto Mask

    btn_fashion_auto_mask.click(fn=run_fashion_auto_mask, inputs=img_fashion_white_bg, outputs=img_fashion_auto_mask
                                ).then(fn=run_fashion_preview_mask, inputs=[img_fashion_white_bg, img_fashion_auto_mask], outputs=img_fashion_auto_mask_preview)

    # Image Segmentation

    btn_fashion_segmentation.click(fn=run_fashion_get_segments, inputs=img_fashion_white_bg, outputs=img_fashion_segments
                                   ).then(fn=create_annotated_image, inputs=[img_fashion_white_bg, img_fashion_segments, clicked_indices],
                                          outputs=[img_fashion_annotated_image_for_mask, clicked_indices])

    img_fashion_annotated_image_for_mask.select(fn=on_annotation_click, inputs=clicked_indices, outputs=[fashion_segment_indices, clicked_indices])

    btn_fashion_mask_preview.click(fn=get_combined_mask, inputs=[img_fashion_segments, fashion_segment_indices, fashion_skip_auto_mask, img_fashion_auto_mask, fashion_state_show_rmbg_retrial], outputs=img_fashion_combined_mask
                                   ).then(fn=run_fashion_preview_mask, inputs=[img_fashion_white_bg, img_fashion_combined_mask], outputs=img_fashion_combined_mask_preview
                                          ).then(fn=run_fashion_postprocess_mask, inputs=img_fashion_combined_mask, outputs=img_fashion_garment_mask
                                                 ).then(fn=run_fashion_whiteBG_garment, inputs=[img_fashion_upscale, img_fashion_garment_mask], outputs=img_fashion_whiteBG_garment)

    # Model Generation

    btn_fashion_model_gen.click(fn=run_fashion_gen_model, inputs=[img_fashion_white_bg, img_fashion_whiteBG_garment, img_fashion_garment_mask, fashion_face_prompt], outputs=img_fashion_model
                                ).then(fn=run_fashion_make_white_bg, inputs=[img_fashion_model], outputs=img_fashion_whiteBG_model
                                       ).then(fn=run_fashion_face_detailer, inputs=[img_fashion_whiteBG_model, img_fashion_upscale, img_fashion_garment_mask, fashion_face_prompt], outputs=img_fashion_with_details)

    # BG Generation

    btn_fashion_bgic.click(fn=run_fashion_change_BG, inputs=[img_fashion_with_details, img_fashion_upscale, img_fashion_garment_mask, fashion_bg_text,
                           fashion_restoration_choices, fashion_lora, fashion_lora_strength, fashion_clip_skip], outputs=img_fashion_changbg_gallery)



