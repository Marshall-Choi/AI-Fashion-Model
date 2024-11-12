# Use Nvidia CUDA base image
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

RUN apt-get update && apt-get install -y \
    cuda-toolkit-12-2

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 

# Install git-lfs for installing large files
RUN apt-get install -y git-lfs
RUN git lfs install

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui

# Change working directory to ComfyUI
WORKDIR /comfyui

# Install ComfyUI dependencies with CUDA 12.1 support
RUN pip3 install --no-cache-dir torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir --no-deps xformers==0.0.26.post1 \
    && pip3 install -r requirements.txt

# Download models
RUN pip install flash-attn

RUN wget -O models/checkpoints/epicrealism_naturalSinRC1VAE.safetensors https://civitai.com/api/download/models/143906

RUN wget -O models/loras/add_detail.safetensors https://civitai.com/api/download/models/62833

RUN wget -O models/controlnet/controlnet++_canny_sd15_fp16.safetensors  https://huggingface.co/huchenlei/ControlNet_plus_plus_collection_fp16/resolve/main/controlnet%2B%2B_canny_sd15_fp16.safetensors
RUN wget -O models/controlnet/controlnet++_depth_sd15_fp16.safetensors  https://huggingface.co/huchenlei/ControlNet_plus_plus_collection_fp16/resolve/main/controlnet%2B%2B_depth_sd15_fp16.safetensors
RUN wget -O models/controlnet/control_v11p_sd15_openpose_fp16.safetensors  https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_openpose-fp16.safetensors
RUN wget -O models/controlnet/control_v11p_sd15_inpaint_fp16.safetensors https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors

RUN mkdir -p models/ella/
RUN mkdir -p models/ella_encoder/models--google--flan-t5-xl--text_encoder/

RUN wget -O models/ella/ella-sd1.5-tsc-t5xl.safetensors https://huggingface.co/QQGYLab/ELLA/resolve/main/ella-sd1.5-tsc-t5xl.safetensors

RUN wget -O models/ella_encoder/models--google--flan-t5-xl--text_encoder/config.json https://huggingface.co/QQGYLab/ELLA/resolve/main/models--google--flan-t5-xl--text_encoder/config.json
RUN wget -O models/ella_encoder/models--google--flan-t5-xl--text_encoder/model.safetensors https://huggingface.co/QQGYLab/ELLA/resolve/main/models--google--flan-t5-xl--text_encoder/model.safetensors
RUN wget -O models/ella_encoder/models--google--flan-t5-xl--text_encoder/special_tokens_map.json https://huggingface.co/QQGYLab/ELLA/resolve/main/models--google--flan-t5-xl--text_encoder/special_tokens_map.json
RUN wget -O models/ella_encoder/models--google--flan-t5-xl--text_encoder/spiece.model https://huggingface.co/QQGYLab/ELLA/resolve/main/models--google--flan-t5-xl--text_encoder/spiece.model
RUN wget -O models/ella_encoder/models--google--flan-t5-xl--text_encoder/tokenizer.json https://huggingface.co/QQGYLab/ELLA/resolve/main/models--google--flan-t5-xl--text_encoder/tokenizer.json
RUN wget -O models/ella_encoder/models--google--flan-t5-xl--text_encoder/tokenizer_config.json https://huggingface.co/QQGYLab/ELLA/resolve/main/models--google--flan-t5-xl--text_encoder/tokenizer_config.json

RUN mkdir -p models/ipadapter/

RUN wget -O models/upscale_models/4x_NMKD-Siax_200k.pth https://huggingface.co/gemasai/4x_NMKD-Siax_200k/resolve/main/4x_NMKD-Siax_200k.pth

RUN wget -O models/vae/vae-ft-mse-840000-ema-pruned.safetensors https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors

RUN mkdir -p models/BiRefNet/
RUN wget -O models/BiRefNet/swin_large_patch4_window12_384_22kto1k.pth https://huggingface.co/ViperYX/BiRefNet/resolve/main/swin_large_patch4_window12_384_22kto1k.pth
RUN wget -O models/BiRefNet/BiRefNet-ep480.pth https://huggingface.co/ViperYX/BiRefNet/resolve/main/BiRefNet-ep480.pth
RUN mkdir -p models/BiRefNet/pth/
RUN wget -O models/BiRefNet/pth/BiRefNet-general-epoch_244.pth https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-epoch_244.pth

RUN mkdir -p models/vitmatte/
RUN wget -O models/vitmatte/config.json https://huggingface.co/hustvl/vitmatte-small-composition-1k/resolve/main/config.json
RUN wget -O models/vitmatte/preprocessor_config.json https://huggingface.co/hustvl/vitmatte-small-composition-1k/resolve/main/preprocessor_config.json
RUN wget -O models/vitmatte/pytorch_model.bin https://huggingface.co/hustvl/vitmatte-small-composition-1k/resolve/main/pytorch_model.bin

RUN mkdir -p models/segformer_b3_fashion
RUN git clone https://huggingface.co/sayeed99/segformer-b3-fashion models/segformer_b3_fashion

RUN mkdir -p models/LLM/
RUN git clone https://huggingface.co/thwri/CogFlorence-2.2-Large models/LLM/CogFlorence-2.2-Large

WORKDIR /comfyui

RUN mkdir -p models/depthanything/

RUN wget -O models/depthanything/depth_anything_v2_vitl_fp16.safetensors https://huggingface.co/Kijai/DepthAnythingV2-safetensors/resolve/main/depth_anything_v2_vitl_fp16.safetensors

RUN mkdir -p models/sam2/ \
    && wget -O models/sam2/sam2_hiera_large.safetensors https://huggingface.co/Kijai/sam2-safetensors/resolve/main/sam2_hiera_large.safetensors

# Download Custom node
RUN git clone https://github.com/TencentQQGYLab/ComfyUI-ELLA.git custom_nodes/ComfyUI-ELLA

RUN git clone https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings.git custom_nodes/Stable-Diffusion-temperature-settings

RUN git clone https://github.com/kijai/ComfyUI-IC-Light.git custom_nodes/ComfyUI-IC-Light

RUN git clone https://github.com/huchenlei/ComfyUI-IC-Light-Native.git custom_nodes/ComfyUI-IC-Light-Native

RUN git clone https://github.com/risunobushi/comfyUI_FrequencySeparation_RGB-HSV.git custom_nodes/comfyUI_FrequencySeparation_RGB-HSV

RUN git clone https://github.com/yolain/ComfyUI-Easy-Use.git custom_nodes/ComfyUI-Easy-Use

RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git custom_nodes/ComfyUI-KJNodes

RUN git clone https://github.com/Derfuu/Derfuu_ComfyUI_ModdedNodes.git custom_nodes/Derfuu_ComfyUI_ModdedNodes

RUN git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git custom_nodes/ComfyUI-Custom-Scripts

RUN git clone https://github.com/rgthree/rgthree-comfy.git custom_nodes/gthree-comfy 

RUN git clone https://github.com/kijai/ComfyUI-DepthAnythingV2.git custom_nodes/ComfyUI-DepthAnythingV2

RUN git clone https://github.com/pionmkh/ComfyUI-Image-Filters.git custom_nodes/ComfyUI-Image-Filters

RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git custom_nodes/ComfyUI-Impact-Pack

RUN git clone https://github.com/Fannovel16/comfyui_controlnet_aux custom_nodes/comfyui_controlnet_aux \
    && mkdir -p custom_nodes/comfyui_controlnet_aux/ckpts \
    && mkdir -p custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel \
    && mkdir -p custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators \
    && wget -O custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators/facenet.pth https://huggingface.co/lllyasviel/Annotators/resolve/main/facenet.pth \
    && wget -O custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators/body_pose_model.pth https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth \
    && wget -O custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators/hand_pose_model.pth https://huggingface.co/lllyasviel/Annotators/resolve/main/hand_pose_model.pth

RUN git clone https://github.com/WASasquatch/was-node-suite-comfyui.git custom_nodes/was-node-suite-comfyui

RUN git clone https://github.com/WASasquatch/WAS_Extras.git custom_nodes/WAS_Extras

RUN git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git custom_nodes/ComfyUI_IPAdapter_plus

RUN git clone https://github.com/Gourieff/comfyui-reactor-node.git custom_nodes/comfyui-reactor-node

RUN git clone https://github.com/Acly/comfyui-tooling-nodes.git custom_nodes/comfyui-tooling-nodes

RUN git clone https://github.com/cubiq/ComfyUI_essentials.git custom_nodes/ComfyUI_essentials

RUN git clone https://github.com/kijai/ComfyUI-Florence2.git custom_nodes/ComfyUI-Florence2

RUN git clone https://github.com/kijai/ComfyUI-segment-anything-2.git custom_nodes/ComfyUI-segment-anything-2

RUN git clone https://github.com/ltdrdata/ComfyUI-Inspire-Pack.git custom_nodes/ComfyUI-Inspire-Pack

RUN git clone https://github.com/BadCafeCode/execution-inversion-demo-comfyui.git custom_nodes/execution-inversion-demo-comfyui

RUN git clone https://github.com/pioncmr/ComfyUI_LayerStyle.git custom_nodes/ComfyUI_LayerStyle 

# Install ComfyUI dependencies

RUN pip3 install -r custom_nodes/ComfyUI_LayerStyle/requirements.txt

RUN pip3 install -r custom_nodes/ComfyUI-ELLA/requirements.txt \
    && pip3 install -r custom_nodes/ComfyUI-Image-Filters/requirements.txt \
    && pip3 install -r custom_nodes/ComfyUI-IC-Light/requirements.txt \
    && pip3 install -r custom_nodes/ComfyUI-Easy-Use/requirements.txt \
    && pip3 install -r custom_nodes/ComfyUI-KJNodes/requirements.txt \
    && pip3 install -r custom_nodes/comfyui-reactor-node/requirements.txt \
    && pip3 install -r custom_nodes/comfyui-tooling-nodes/requirements.txt \
    && pip3 install -r custom_nodes/ComfyUI_essentials/requirements.txt \
    && pip3 install -r custom_nodes/ComfyUI-Florence2/requirements.txt \
    && pip3 install -r custom_nodes/ComfyUI-segment-anything-2/requirements.txt \
    && pip3 install -r custom_nodes/ComfyUI-Impact-Pack/requirements.txt \
    && pip3 install -r custom_nodes/ComfyUI-Inspire-Pack/requirements.txt 

RUN python3 custom_nodes/ComfyUI-Impact-Pack/install.py

# Install runpod
RUN pip3 install runpod requests

RUN pip3 uninstall -y opencv-python opencv-python-headless opencv-contrib-python-headless opencv-contrib-python \
    && pip3 install opencv-python opencv-python-headless opencv-contrib-python-headless \
    && pip3 install opencv-contrib-python

RUN pip3 install pilgram
RUN pip3 cache purge

RUN git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git custom_nodes/ComfyUI_Comfyroll_CustomNodes

# Support for the network volume
COPY src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Add ComfyUI workflow
COPY /src/workflow/ ./workflow/

# Add the start and the handler
COPY src/start.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh

# Start the container
CMD /start.sh
