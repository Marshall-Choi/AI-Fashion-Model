# Use Nvidia CUDA base image
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

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
    libglib2.0-0 \
    libegl1-mesa \
    libgles2-mesa \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libgl1-mesa-dev \
    libx11-dev \
    xvfb libxext-dev

# Install mediapipe
    
RUN pip3 install --no-cache-dir mediapipe

# Install git-lfs for installing large files
RUN apt-get install -y git-lfs
RUN git lfs install

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui

# Change working directory to ComfyUI
WORKDIR /comfyui

# Install ComfyUI dependencies with CUDA 12.4 support
RUN pip3 install --no-cache-dir torch==2.5.1+cu124 torchvision==0.20.1+cu124 xformers==0.0.29 --extra-index-url https://download.pytorch.org/whl/cu124 \
    && pip3 install -r requirements.txt

RUN apt-get update && apt-get install -y curl && apt-get clean

# Download models

# Checkpoints
RUN wget -O models/checkpoints/epicphotogasm.safetensors https://civitai.com/api/download/models/429454
RUN curl -L \
-H "Content-Type: application/json" \
-H "Authorization: Bearer a0e42c3983bc134833695ad26affbe6d" \
"https://civitai.com/api/download/models/920957" --output models/checkpoints/JuggernautXL.safetensors

# LoRAs
RUN wget -O models/loras/SOAP.safetensors https://civitai.com/api/download/models/144929

# SDXL controlnet
RUN mkdir -p models/controlnet/ \
    && wget -O models/controlnet/controlnet-union-sdxl-1.0.promax.safetensors https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promax.safetensors \
    && wget -O models/controlnet/OpenPoseXL2.safetensors https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0/resolve/main/OpenPoseXL2.safetensors

# SDXL EcomID
RUN mkdir -p models/pulid/ \
    && wget -O models/pulid/ip-adapter_pulid_sdxl_fp16.safetensors https://huggingface.co/huchenlei/ipadapter_pulid/resolve/main/ip-adapter_pulid_sdxl_fp16.safetensors 

RUN mkdir -p models/insightface/ \
    && mkdir -p models/insightface/models/ \
    && mkdir -p models/insightface/models/antelopev2/ \
    && wget -O models/insightface/models/antelopev2/1k3d68.onnx https://huggingface.co/MonsterMMORPG/tools/resolve/main/1k3d68.onnx \
    && wget -O models/insightface/models/antelopev2/2d106det.onnx https://huggingface.co/MonsterMMORPG/tools/resolve/main/2d106det.onnx \
    && wget -O models/insightface/models/antelopev2/genderage.onnx https://huggingface.co/MonsterMMORPG/tools/resolve/main/genderage.onnx \
    && wget -O models/insightface/models/antelopev2/glintr100.onnx https://huggingface.co/MonsterMMORPG/tools/resolve/main/glintr100.onnx \
    && wget -O models/insightface/models/antelopev2/scrfd_10g_bnkps.onnx https://huggingface.co/MonsterMMORPG/tools/resolve/main/scrfd_10g_bnkps.onnx \
    && wget -O models/insightface/inswapper_128.onnx https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx

RUN mkdir -p models/instantid/ \
    && wget -O models/instantid/ip-adapter.bin https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin

RUN wget -O models/controlnet/controlnet_ecomid_sdxl.safetensors https://huggingface.co/alimama-creative/SDXL-EcomID/resolve/main/diffusion_pytorch_model.safetensors

# ReActor
RUN mkdir -p models/facerestore_models/ \
    && wget -O models/facerestore_models/GPEN-BFR-1024.onnx https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-1024.onnx

# SD 1.5 controlnet
RUN wget -O models/controlnet/control_sd15_inpaint_depth_hand_fp16.safetensors https://huggingface.co/hr16/ControlNet-HandRefiner-pruned/resolve/main/control_sd15_inpaint_depth_hand_fp16.safetensors

RUN mkdir -p models/ipadapter/

RUN wget -O models/upscale_models/4x_NMKD-Siax_200k.pth https://huggingface.co/gemasai/4x_NMKD-Siax_200k/resolve/main/4x_NMKD-Siax_200k.pth

RUN mkdir -p models/BiRefNet/
RUN wget -O models/BiRefNet/swin_large_patch4_window12_384_22kto1k.pth https://huggingface.co/ViperYX/BiRefNet/resolve/main/swin_large_patch4_window12_384_22kto1k.pth
RUN wget -O models/BiRefNet/BiRefNet-ep480.pth https://huggingface.co/ViperYX/BiRefNet/resolve/main/BiRefNet-ep480.pth
RUN mkdir -p models/BiRefNet/pth/
RUN wget -O models/BiRefNet/pth/BiRefNet-general-epoch_244.pth https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-epoch_244.pth

RUN mkdir -p models/vitmatte/
RUN wget -O models/vitmatte/config.json https://huggingface.co/hustvl/vitmatte-small-composition-1k/resolve/main/config.json
RUN wget -O models/vitmatte/preprocessor_config.json https://huggingface.co/hustvl/vitmatte-small-composition-1k/resolve/main/preprocessor_config.json
RUN wget -O models/vitmatte/model.safetensors https://huggingface.co/hustvl/vitmatte-small-composition-1k/resolve/main/model.safetensors

RUN mkdir -p models/segformer_b3_fashion
RUN git clone https://huggingface.co/sayeed99/segformer-b3-fashion models/segformer_b3_fashion

RUN mkdir -p models/LLM/
RUN git clone https://huggingface.co/thwri/CogFlorence-2.2-Large models/LLM/CogFlorence-2.2-Large

RUN mkdir -p models/depthanything/

RUN wget -O models/depthanything/depth_anything_v2_vitl_fp16.safetensors https://huggingface.co/Kijai/DepthAnythingV2-safetensors/resolve/main/depth_anything_v2_vitl_fp16.safetensors

RUN mkdir -p models/sam2/ \
    && wget -O models/sam2/sam2_hiera_large.safetensors https://huggingface.co/Kijai/sam2-safetensors/resolve/main/sam2_hiera_large.safetensors

RUN mkdir -p models/ultralytics/ && mkdir -p models/ultralytics/bbox/ \
    && wget -O models/ultralytics/bbox/face_yolov8m.pt https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/detection/bbox/face_yolov8m.pt \
    && wget -O models/ultralytics/bbox/hand_yolov8s.pt https://huggingface.co/xingren23/comfyflow-models/resolve/976de8449674de379b02c144d0b3cfa2b61482f2/ultralytics/bbox/hand_yolov8s.pt

RUN pip install facexlib
RUN wget -q https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth -O /usr/local/lib/python3.10/dist-packages/facexlib/weights/detection_Resnet50_Final.pth
RUN wget -q https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth -O /usr/local/lib/python3.10/dist-packages/facexlib/weights/parsing_parsenet.pth
RUN wget -q https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth -O /usr/local/lib/python3.10/dist-packages/facexlib/weights/parsing_bisenet.pth

RUN mkdir -p models/insightface/models/buffalo_l/ \
    && wget -O models/insightface/models/buffalo_l/1k3d68.onnx https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/1k3d68.onnx \
    && wget -O models/insightface/models/buffalo_l/2d106det.onnx https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/2d106det.onnx \
    && wget -O models/insightface/models/buffalo_l/det_10g.onnx https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/det_10g.onnx \
    && wget -O models/insightface/models/buffalo_l/genderage.onnx https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/genderage.onnx \
    && wget -O models/insightface/models/buffalo_l/w600k_r50.onnx https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx 

# Download Custom node
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
    && wget -O custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators/hand_pose_model.pth https://huggingface.co/lllyasviel/Annotators/resolve/main/hand_pose_model.pth \
    && mkdir -p custom_nodes/comfyui_controlnet_aux/ckpts/hr16 \
    && mkdir -p custom_nodes/comfyui_controlnet_aux/ckpts/hr16/yolo-nas-fp16 \
    && mkdir -p custom_nodes/comfyui_controlnet_aux/ckpts/yzd-v \
    && mkdir -p custom_nodes/comfyui_controlnet_aux/ckpts/yzd-v/DWPose \
    && wget -O custom_nodes/comfyui_controlnet_aux/ckpts/hr16/yolo-nas-fp16/yolo_nas_l_fp16.onnx https://huggingface.co/hr16/yolo-nas-fp16/resolve/main/yolo_nas_l_fp16.onnx \
    && wget -O custom_nodes/comfyui_controlnet_aux/ckpts/yzd-v/DWPose/dw-ll_ucoco_384.onnx https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx \
    && mkdir -p custom_nodes/comfyui_controlnet_aux/ckpts/hr16/ControlNet-HandRefiner-pruned \
    && wget -O custom_nodes/comfyui_controlnet_aux/ckpts/hr16/ControlNet-HandRefiner-pruned/graphormer_hand_state_dict.bin https://huggingface.co/hr16/ControlNet-HandRefiner-pruned/resolve/main/graphormer_hand_state_dict.bin \
    && wget -O custom_nodes/comfyui_controlnet_aux/ckpts/hr16/ControlNet-HandRefiner-pruned/hrnetv2_w64_imagenet_pretrained.pth https://huggingface.co/hr16/ControlNet-HandRefiner-pruned/resolve/main/hrnetv2_w64_imagenet_pretrained.pth 
    
RUN git clone https://github.com/WASasquatch/was-node-suite-comfyui.git custom_nodes/was-node-suite-comfyui

RUN git clone https://github.com/WASasquatch/WAS_Extras.git custom_nodes/WAS_Extras

RUN git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git custom_nodes/ComfyUI_IPAdapter_plus

RUN git clone https://github.com/Navezjt/comfyui-reactor-node.git custom_nodes/comfyui-reactor-node

RUN git clone https://github.com/Acly/comfyui-tooling-nodes.git custom_nodes/comfyui-tooling-nodes

RUN git clone https://github.com/cubiq/ComfyUI_essentials.git custom_nodes/ComfyUI_essentials

RUN git clone https://github.com/kijai/ComfyUI-Florence2.git custom_nodes/ComfyUI-Florence2

RUN git clone https://github.com/kijai/ComfyUI-segment-anything-2.git custom_nodes/ComfyUI-segment-anything-2

RUN git clone https://github.com/ltdrdata/ComfyUI-Inspire-Pack.git custom_nodes/ComfyUI-Inspire-Pack

RUN git clone https://github.com/BadCafeCode/execution-inversion-demo-comfyui.git custom_nodes/execution-inversion-demo-comfyui

RUN git clone https://github.com/chflame163/ComfyUI_LayerStyle.git custom_nodes/ComfyUI_LayerStyle 

RUN git clone https://github.com/chflame163/ComfyUI_LayerStyle_Advance.git custom_nodes/ComfyUI_LayerStyle_Advance \
    && cd custom_nodes/ComfyUI_LayerStyle_Advance \
    && git checkout 1b52c5d89bb4bee0969fc9aca615afcf9c283f4a
RUN git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git custom_nodes/ComfyUI_Comfyroll_CustomNodes

RUN git clone https://github.com/alimama-creative/SDXL_EcomID_ComfyUI.git custom_nodes/SDXL_EcomID_ComfyUI

RUN git clone https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git custom_nodes/ComfyUI-Advanced-ControlNet

RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git custom_nodes/ComfyUI-Impact-Subpack \
    && cd custom_nodes/ComfyUI-Impact-Subpack \
    && git checkout 8628fa3a8abf168326afc0ade9666802ca3a0e86

# Install ComfyUI dependencies

RUN pip3 install -r custom_nodes/ComfyUI_LayerStyle/requirements.txt \
    && pip3 install -r custom_nodes/ComfyUI_LayerStyle_Advance/requirements.txt

RUN pip3 install -r custom_nodes/SDXL_EcomID_ComfyUI/requirements.txt \
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
    && pip3 install -r custom_nodes/ComfyUI-Impact-Subpack/requirements.txt \
    && pip3 install -r custom_nodes/ComfyUI-Inspire-Pack/requirements.txt \
    && pip3 install -r custom_nodes/comfyui_controlnet_aux/requirements.txt

RUN python3 custom_nodes/ComfyUI-Impact-Pack/install.py

# Install runpod
RUN pip3 install runpod requests

RUN pip3 uninstall -y opencv-python opencv-python-headless opencv-contrib-python-headless opencv-contrib-python \
    && pip3 install opencv-python opencv-python-headless opencv-contrib-python-headless \
    && pip3 install opencv-contrib-python

RUN pip3 install pilgram
RUN pip3 cache purge

# Clean up output directory
RUN rm -f output/*.png
RUN mkdir models/nsfw_detector \
    && mkdir models/nsfw_detector/vit-base-nsfw-detector \
    && wget -O models/nsfw_detector/vit-base-nsfw-detector/config.json https://huggingface.co/AdamCodd/vit-base-nsfw-detector/resolve/main/config.json \
    && wget -O models/nsfw_detector/vit-base-nsfw-detector/model.safetensors https://huggingface.co/AdamCodd/vit-base-nsfw-detector/resolve/main/model.safetensors \
    && wget -O models/nsfw_detector/vit-base-nsfw-detector/preprocessor_config.json https://huggingface.co/AdamCodd/vit-base-nsfw-detector/resolve/main/preprocessor_config.json
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
