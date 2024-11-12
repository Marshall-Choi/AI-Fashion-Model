import runpod
from runpod.serverless.utils import rp_upload
import json
import urllib.request
import urllib.parse
import time
import os
import requests
import base64
import cv2
import numpy as np
from io import BytesIO

# Time to wait between API check attempts in milliseconds
COMFY_API_AVAILABLE_INTERVAL_MS = 250
# Maximum number of API check attempts
COMFY_API_AVAILABLE_MAX_RETRIES = 500
# Time to wait between poll attempts in milliseconds
COMFY_POLLING_INTERVAL_MS = os.environ.get("COMFY_POLLING_INTERVAL_MS", 250)
# Maximum number of poll attempts
COMFY_POLLING_MAX_RETRIES = os.environ.get("COMFY_POLLING_MAX_RETRIES", 500)
# Host where ComfyUI is running
COMFY_HOST = "127.0.0.1:8188"
# Enforce a clean state after each job is done
# see https://docs.runpod.io/docs/handler-additional-controls#refresh-worker
REFRESH_WORKER = os.environ.get("REFRESH_WORKER", "false").lower() == "true"

output_name_select = None

def validate_input(job_input):
    """
    Validates the input for the handler function.

    Args:
        job_input (dict): The input data to validate.

    Returns:
        tuple: A tuple containing the validated data and an error message, if any.
               The structure is (validated_data, error_message).
    """
    # Validate if job_input is provided
    if job_input is None:
        return None, "Please provide input"

    # Check if input is a string and try to parse it as JSON
    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, "Invalid JSON format in input"

    # # Validate 'command' in input
    command = job_input.get("command")
    try:
        with open(f'./workflow/{command}.json', 'r') as file:
            workflow = json.load(file)
    except:
        return None, "Invalid 'command' parameter"

    # Validate 'images' in input, if provided
    images = job_input.get("image_urls")
    id_to_class_type = {id: details.get('_meta').get('title') for id, details in workflow.items()}
    if images is not None:
        if not isinstance(images, list):
            return (
                None, "'image_urls' must be a list of image_urls",
            )

        msk_done = 0
        rbg_done = 0
        dth_done = 0
        img_done = 0

        for image_url in images:
            if image_url.startswith("http://") or image_url.startswith("https://"):
                response = requests.get(image_url, timeout=5)
                if response.status_code != 200:
                    return None, f"Url server's status_code is not 200: {image_url}"
                img = base64.b64encode(response.content).decode("utf-8")
            elif image_url.startswith("data:image/"):
                img = image_url.split(",")[1]

            loadmsk_id = [key for key, value in id_to_class_type.items() if value == 'Load Mask (Base64)']
            if loadmsk_id:
                workflow.get(f'{loadmsk_id[msk_done]}').get('inputs')['mask'] = img
                msk_done += 1
                continue

            loadrbg_id = [key for key, value in id_to_class_type.items() if value == 'Load Image (Base64) RMBG']
            if loadrbg_id:
                workflow.get(f'{loadrbg_id[rbg_done]}').get('inputs')['base64_data'] = img
                rbg_done += 1
                continue

            loaddth_id = [key for key, value in id_to_class_type.items() if value == 'Load Image (Base64) Depth']
            if loaddth_id:
                workflow.get(f'{loaddth_id[dth_done]}').get('inputs')['base64_data'] = img
                dth_done += 1
                continue

            loadimg_id = [key for key, value in id_to_class_type.items() if value == 'Load Image (Base64)']
            if loadimg_id:
                workflow.get(f'{loadimg_id[img_done]}').get('inputs')['base64_data'] = img
                img_done += 1
                continue
            
    prompt = job_input.get('prompt')
    if prompt is not None:
        prompt_id = [key for key, value in id_to_class_type.items() if value == 'Prompt']
        if prompt_id:
            workflow.get(f'{prompt_id[0]}').get('inputs')['Text'] = prompt
            print("Got the prompt!: ", workflow.get(f'{prompt_id[0]}').get('inputs')['Text'])

    restoration_choice = job_input.get('restoration_choices')
    global output_name_select
    if restoration_choice is not None:
        choice_id = [key for key, value in id_to_class_type.items() if value == 'restoration_choice'][0]
        workflow.get(choice_id).get('inputs')['string'] = ' '.join(restoration_choice)
        output_name_select = restoration_choice
    else:
        output_name_select = None
    
    lora_setting = job_input.get('lora_setting')
    if lora_setting is not None:
        lora_id = [key for key, value in id_to_class_type.items() if value == 'Load LoRA'][0]
        lora = lora_setting.get('lora')
        if lora != 'Not use':
            workflow.get(lora_id).get('inputs')['lora_name'] = 'add_detail.safetensors' if 'v1' else 'detailSliderALT2.safetensors'
            workflow.get(lora_id).get('inputs')['strength_model'] = lora_setting.get('strength')
        else :
            workflow.get(lora_id).get('inputs')['strength_model'] = 0
            workflow.get(choice_id).get('inputs')['strength_clip'] = 0
    
    clip_skip = job_input.get('clip_skip')
    if clip_skip is not None:
        clip_id = [key for key, value in id_to_class_type.items() if value == 'CLIP Set Last Layer'][0]
        workflow.get(lora_id).get('inputs')['stop_at_clip_layer'] = -2 if clip_skip else -1

    fashion_item = job_input.get('fashion_item')
    if fashion_item is not None:
        florence_id = [key for key, value in id_to_class_type.items() if value == 'Florence2Run'][0]
        print("florence id: ", florence_id)
        workflow.get(f'{florence_id}').get('inputs')['text_input'] = fashion_item

    model_prompt = job_input.get('model_prompt')
    if model_prompt is not None:
        model_prompt_id = [key for key, value in id_to_class_type.items() if value == 'Model Prompt']
        if model_prompt_id:
            workflow.get(f'{model_prompt_id[0]}').get('inputs')['text'] = model_prompt
            print("Got the prompt!: ", workflow.get(f'{model_prompt_id[0]}').get('inputs')['text'])

    wildcard = job_input.get('wildcard')
    if wildcard is not None:
        wildcard_id = [key for key, value in id_to_class_type.items() if value == 'FaceDetailer']
        if wildcard_id:
            workflow.get(f'{wildcard_id[0]}').get('inputs')['wildcard'] = wildcard

    use_FD = job_input.get('use_FD')
    if use_FD is not None:
        use_FD_id = [key for key, value in id_to_class_type.items() if value == 'Boolean_FD']
        if use_FD_id:
            workflow.get(f'{use_FD_id[0]}').get('inputs')['value'] = use_FD

    # Return validated data and no error
    return {"workflow": workflow}, None


def check_server(url, retries=500, delay=50):
    """
    Check if a server is reachable via HTTP GET request

    Args:
    - url (str): The URL to check
    - retries (int, optional): The number of times to attempt connecting to the server. Default is 50
    - delay (int, optional): The time in milliseconds to wait between retries. Default is 500

    Returns:
    bool: True if the server is reachable within the given number of retries, otherwise False
    """

    for i in range(retries):
        try:
            response = requests.get(url)

            # If the response status code is 200, the server is up and running
            if response.status_code == 200:
                print(f"runpod-worker-comfy - API is reachable")
                return True
        except requests.RequestException as e:
            # If an exception occurs, the server may not be ready
            pass

        # Wait for the specified delay before retrying
        time.sleep(delay / 1000)

    print(
        f"runpod-worker-comfy - Failed to connect to server at {url} after {retries} attempts."
    )
    return False


# def upload_images(images):
#     """
#     Upload a list of base64 encoded images to the ComfyUI server using the /upload/image endpoint.

#     Args:
#         images (list): A list of dictionaries, each containing the 'name' of the image and the 'image' as a base64 encoded string.
#         server_address (str): The address of the ComfyUI server.

#     Returns:
#         list: A list of responses from the server for each image upload.
#     """
#     if not images:
#         return {"status": "success", "message": "No images to upload", "details": []}

#     responses = []
#     upload_errors = []

#     print(f"runpod-worker-comfy - image(s) upload")

#     for image in images:
#         name = image["name"]
#         image_data = image["image"]
#         blob = base64.b64decode(image_data)

#         # Prepare the form data
#         files = {
#             "image": (name, BytesIO(blob), "image/png"),
#             "overwrite": (None, "true"),
#         }

#         # POST request to upload the image
#         response = requests.post(f"http://{COMFY_HOST}/upload/image", files=files)
#         if response.status_code != 200:
#             upload_errors.append(f"Error uploading {name}: {response.text}")
#         else:
#             responses.append(f"Successfully uploaded {name}")

#     if upload_errors:
#         print(f"runpod-worker-comfy - image(s) upload with errors")
#         return {
#             "status": "error",
#             "message": "Some images failed to upload",
#             "details": upload_errors,
#         }

#     print(f"runpod-worker-comfy - image(s) upload complete")
#     return {
#         "status": "success",
#         "message": "All images uploaded successfully",
#         "details": responses,
#     }


def queue_workflow(workflow):
    """
    Queue a workflow to be processed by ComfyUI

    Args:
        workflow (dict): A dictionary containing the workflow to be processed

    Returns:
        dict: The JSON response from ComfyUI after processing the workflow
    """

    # The top level element "prompt" is required by ComfyUI
    data = json.dumps({"prompt": workflow}).encode("utf-8")

    req = urllib.request.Request(f"http://{COMFY_HOST}/prompt", data=data)
    return json.loads(urllib.request.urlopen(req).read())


def get_history(prompt_id):
    """
    Retrieve the history of a given prompt using its ID

    Args:
        prompt_id (str): The ID of the prompt whose history is to be retrieved

    Returns:
        dict: The history of the prompt, containing all the processing steps and results
    """
    with urllib.request.urlopen(f"http://{COMFY_HOST}/history/{prompt_id}") as response:
        result = json.loads(response.read())
        return result


def base64_encode(img_path):
    """
    Returns base64 encoded image.

    Args:
        img_path (str): The path to the image

    Returns:
        str: The base64 encoded image
    """
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"{encoded_string}"


def process_output_images(outputs, job_id):
    """
    This function takes the "outputs" from image generation and the job ID,
    then determines the correct way to return the image, either as a direct URL
    to an AWS S3 bucket or as a base64 encoded string, depending on the
    environment configuration.

    Args:
        outputs (dict): A dictionary containing the outputs from image generation,
                        typically includes node IDs and their respective output data.
        job_id (str): The unique identifier for the job.

    Returns:
        dict: A dictionary with the status ('success' or 'error') and the message,
              which is either the URL to the image in the AWS S3 bucket or a base64
              encoded string of the image. In case of error, the message details the issue.

    The function works as follows:
    - It first determines the output path for the images from an environment variable,
      defaulting to "/comfyui/output" if not set.
    - It then iterates through the outputs to find the filenames of the generated images.
    - After confirming the existence of the image in the output folder, it checks if the
      AWS S3 bucket is configured via the BUCKET_ENDPOINT_URL environment variable.
    - If AWS S3 is configured, it uploads the image to the bucket and returns the URL.
    - If AWS S3 is not configured, it encodes the image in base64 and returns the string.
    - If the image file does not exist in the output folder, it returns an error status
      with a message indicating the missing image file.
    """

    # The path where ComfyUI stores the generated images
    COMFY_OUTPUT_PATH = os.environ.get("COMFY_OUTPUT_PATH", "/comfyui/output")

    print(f"runpod-worker-comfy - image generation is done")
    all_images_base64 = []
    text = ""
    for node_id, node_output in outputs.items():
        if "images" in node_output:
            for image in node_output.get("images"):
                output_image_path = os.path.join(image.get('subfolder'), image.get('filename'))

                # expected image output folder
                local_image_path = f"{COMFY_OUTPUT_PATH}/{output_image_path}"

                print(f"runpod-worker-comfy - {local_image_path}")

                # The image is in the output folder
                if os.path.exists(local_image_path):
                    # base64 image
                    image_base64 = base64_encode(local_image_path)
                    print("runpod-worker-comfy - the image was generated and converted to base64")

                    # Add the base64 image to the list
                    image_dict = {
                        'url': 'data:image/png;base64,' + image_base64,
                    }
                    if output_name_select is not None:
                        image_dict.update({'restoration_choice': output_name_select.pop(0)})
                    all_images_base64.append(image_dict)
                else:
                    print("runpod-worker-comfy - the image does not exist in the output folder")
                    return {
                        "status": "error",
                        "message": f"the image does not exist in the specified output folder: {local_image_path}",
                    }
        if "text" in node_output:
            text = node_output.get("text")[0]
    result = {
        "status": "success",
        "outputs": all_images_base64,  
        "aux": text
    }

    return result

def handler(job):
    """
    The main function that handles a job of generating an image.

    This function validates the input, sends a prompt to ComfyUI for processing,
    polls ComfyUI for result, and retrieves generated images.

    Args:
        job (dict): A dictionary containing job details and input parameters.

    Returns:
        dict: A dictionary containing either an error message or a success status with generated images.
    """
    job_input = job["input"]

    # Make sure that the input is valid
    validated_data, error_message = validate_input(job_input)
    if error_message:
        return {"error": error_message}

    # Extract validated data
    workflow = validated_data["workflow"]

    # Make sure that the ComfyUI API is available
    check_server(
        f"http://{COMFY_HOST}",
        COMFY_API_AVAILABLE_MAX_RETRIES,
        COMFY_API_AVAILABLE_INTERVAL_MS,
    )

    # Queue the workflow
    try:
        queued_workflow = queue_workflow(workflow)
        prompt_id = queued_workflow["prompt_id"]
        print(f"runpod-worker-comfy - queued workflow with ID {prompt_id}")
    except Exception as e:
        return {"error": f"Error queuing workflow: {str(e)}"}

    # Poll for completion
    print(f"runpod-worker-comfy - wait until image generation is complete")
    retries = 0
    try:
        while retries < COMFY_POLLING_MAX_RETRIES:
            history = get_history(prompt_id)

            # Exit the loop if we have found the history
            if prompt_id in history and history[prompt_id].get("outputs"):
                break
            else:
                # Wait before trying again
                time.sleep(COMFY_POLLING_INTERVAL_MS / 1000)
                retries += 1
        else:
            return {"error": "Max retries reached while waiting for image generation"}
    except Exception as e:
        return {"error": f"Error waiting for image generation: {str(e)}"}
    comfyui_status = history[prompt_id]["status"]
    if comfyui_status["status_str"] == "error":
        images_result = {
            "status" : "error",
            "message" : comfyui_status["messages"][2][1]["exception_message"],
            "details" : comfyui_status["messages"][2][1]["traceback"]
        }
    else :
        # Get the generated image and return it as URL in an AWS bucket or as base64
        images_result = process_output_images(history[prompt_id].get("outputs"), job["id"])

    result = {**images_result, "refresh_worker": REFRESH_WORKER}
    
    return result


# Start the handler only if this script is run directly
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
