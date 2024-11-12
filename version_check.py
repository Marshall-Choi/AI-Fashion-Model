import subprocess
import re

# 검색할 패키지 목록
package_list = [
    "numpy", "pillow", "torch", "matplotlib", "Scipy", "scikit_image", "scikit_learn",
    "opencv-contrib-python", "pymatting", "segment_anything", "timm", "addict", "yapf",
    "colour-science", "wget", "mediapipe", "loguru", "typer_config", "fastapi", "rich",
    "google-generativeai", "diffusers", "omegaconf", "tqdm", "kornia", "ultralytics",
    "blend_modes", "blind-watermark", "qrcode", "pyzbar", "transparent-background",
    "huggingface_hub", "accelerate", "onnxruntime", "torchscale", "wandb",
    "psd-tools", "hydra-core", "protobuf>=3.20.3", "inference-cli",
    "inference-gpu[yolo-world]", "bitsandbytes", "transformers",
    "peft", "iopath"
]

# pip list 결과를 가져와서 출력
pip_list_output = subprocess.check_output(["pip", "list"]).decode("utf-8")
installed_packages = dict(re.findall(r"(\S+)\s+(\S+)", pip_list_output))

# 파일로 결과 저장
with open("installed_packages.txt", "w") as file:
    for package in package_list:
        pkg_name = re.split(r"[<>=]", package)[0]
        version = installed_packages.get(pkg_name, "Not Installed")
        file.write(f"{pkg_name}: {version}\n")

print("패키지 버전 정보가 'installed_packages.txt' 파일에 저장되었습니다.")
