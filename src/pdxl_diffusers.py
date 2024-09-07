import modal
import subprocess
import shutil
import os
import urllib.request
import requests
import io
import base64


app = modal.App("pony-diffusion-v6-xl-v4")
secret = modal.Secret.from_name(
    "r2-aws-secret",
    required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
)


image = (
    modal.Image.debian_slim(python_version="3.12.5")
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    )
    .pip_install(
        "diffusers==0.30.2",
        "invisible_watermark==0.2.0",
        "transformers~=4.38.2",
        "accelerate==0.33.0",
        "safetensors==0.4.4",
    )
)

with image.imports():
    import torch
    from diffusers import StableDiffusionPipeline
    from fastapi import Response

MODEL_URL = "https://civitai.com/api/download/models/290640"
MODEL_FILE_PATH = "/checkpoints/pony-diffusion-v6-xl.safetensors"

@app.cls(
    gpu="A10G",
    container_idle_timeout=120,
    image=image,
    volumes={
        "/r2": modal.CloudBucketMount(
        bucket_name="mac-remote",
        bucket_endpoint_url="https://94576bd2d07ef652cc3519ae617a3856.r2.cloudflarestorage.com",
        secret=secret,
        read_only=True
    )
})
class Model:
    @modal.build()
    def build(self):
        print("🛠️ Building container...")

        os.makedirs(os.path.dirname(MODEL_FILE_PATH), exist_ok=True)

        with open(MODEL_FILE_PATH, "wb") as file:
            file.write(requests.get(MODEL_URL).content)

        # download_url = 'https://civitai.com/api/download/models/710901'
        # destination_file = f"/checkpoints/{MODEL_FILE_NAME}"

        # # Download the file with progress monitoring
        # def download_progress(count, block_size, total_size):
        #     percent = int(count * block_size * 100 / total_size)
        #     print(f"Downloading: {percent}%", end="\r")

        # urllib.request.urlretrieve(download_url, destination_file)
        # print("Download complete.")

        # print("☁️ r2 checkpoints folder:")
        # subprocess.run(["ls", "/r2/models/checkpoints"])

        # src_file = f'/r2/models/checkpoints/{MODEL_FILE_NAME}'
        # dst_file = f'/checkpoints/{MODEL_FILE_NAME}'

        # # Create the destination directory if it doesn't exist
        # dst_dir = os.path.dirname(dst_file)
        # os.makedirs(dst_dir, exist_ok=True)

        # # Copy the file
        # shutil.copy(src_file, dst_file)

        # print("🤖 Looking up checkpoints folder...")
        # subprocess.run(["ls", "/checkpoints"])


    @modal.enter()
    def enter(self):
        print("✅ Entering model...")

    @modal.exit()
    def exit(self):
        print("🧨 Exiting model...")

    @modal.method()
    def generate_image(self, prompt:str):
        pipeline = StableDiffusionPipeline.from_single_file(
            MODEL_FILE_PATH ,
            torch_dtype=torch.float16,
            useSafetensors=True,
            local_files_only=True
        )
        pipeline.to("cuda")
        print ("🔥 PIPELINE IS:", pipeline)

        # image = pipeline("An image of a squirrel in Picasso style").images[0]

        image = pipeline(
            prompt=prompt,
            negative_prompt="bad quality",
            num_inference_steps=20,
            denoising_end=0.8,
            width=1024,
            height=1024,
        ).images[0]
        print("🔥 IMAGE IS:", image)

        # Save the generated image
        byte_stream = io.BytesIO()
        image.save(byte_stream, format="PNG")
        img_str = base64.b64encode(byte_stream.getvalue()).decode()

        print(f"Base64 string of the image: {img_str}")  # Print first 50 characters as preview

        # image.save(byte_stream, format="JPEG",)
        # print("Image generated and saved as 'generated_image.png'.")


@app.local_entrypoint()
def main(prompt: str):
    Model().generate_image.remote(prompt)
