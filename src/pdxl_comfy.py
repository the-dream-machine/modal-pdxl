import modal
import os
import requests
import io
import subprocess
# import base64

app = modal.App("pony-diffusion")

image = (
    modal.Image.debian_slim(python_version="3.12.5")
    .apt_install("git", "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1")
    .pip_install("torch==2.4.0+cu121", "torchvision", extra_options="--index-url https://download.pytorch.org/whl/cu121")
    .pip_install("xformers==0.0.27.post2")
    .pip_install("git+https://github.com/hiddenswitch/ComfyUI.git", extra_options="--no-build-isolation")
    .run_commands("comfyui --create-directories")
    .pip_install("comfy-script[default]", extra_options="--upgrade")
)

MODEL_URL = "https://civitai.com/api/download/models/290640"
MODEL_FILE_NAME = "pony-diffusion-v6-xl.safetensors"
MODEL_FILE_PATH = f"/models/checkpoints/{MODEL_FILE_NAME}"

@app.cls(gpu="T4", container_idle_timeout=120, image=image)
class Model:
    @modal.build()
    def build(self):
        print("🛠️ Building container...")

        os.makedirs(os.path.dirname(MODEL_FILE_PATH), exist_ok=True)
        with open(MODEL_FILE_PATH, "wb") as file:
            _ = file.write(requests.get(MODEL_URL).content)

    @modal.enter()
    def enter(self):
        print("✅ Entering container...")
        from comfy_script.runtime import load
        load("comfyui")


    @modal.exit()
    def exit(self):
        print("🧨 Exiting container...")

    @modal.method()
    def generate_image(self, prompt:str):
        print("🎨 Generating image...")

        from comfy_script.runtime import Workflow
        from comfy_script.runtime.nodes import CheckpointLoaderSimple, CLIPTextEncode, EmptyLatentImage, KSampler, VAEDecode, SaveImage

        with Workflow(wait=True):
            model, clip, vae = CheckpointLoaderSimple(MODEL_FILE_NAME)
            conditioning = CLIPTextEncode('beautiful scenery nature glass bottle landscape, , purple galaxy bottle,', clip)
            conditioning2 = CLIPTextEncode('text, watermark', clip)
            latent = EmptyLatentImage(512, 512, 1)
            latent = KSampler(model, 156680208700286, 20, 8, 'euler', 'normal', conditioning, conditioning2, latent, 1)
            image = VAEDecode(latent, vae)
            SaveImage(image, 'ComfyUI')

@app.local_entrypoint()
def main(prompt: str):
    Model().generate_image.remote(prompt)
