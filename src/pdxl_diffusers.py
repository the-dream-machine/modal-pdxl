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
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

    # from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL


    # pipeline = StableDiffusionXLPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    # ).to("cuda")

MODEL_URL = "https://civitai.com/api/download/models/290640?type=Model&format=SafeTensor&size=pruned&fp=fp16"
MODEL_FILE_PATH = "/checkpoints/pony-diffusion-v6-xl.safetensors"

VAE_URL="https://civitai.com/api/download/models/290640?type=VAE&format=SafeTensor"
VAE_FILE_PATH="/vae/pony-diffusion-v6-xl-vae.safetensors"

@app.cls(
    gpu="T4",
    container_idle_timeout=120,
    image=image,
    volumes={
        "/r2": modal.CloudBucketMount(
        bucket_name="mac-remote",
        bucket_endpoint_url="https://94576bd2d07ef652cc3519ae617a3856.r2.cloudflarestorage.com",
        secret=secret,
        # read_only=True
    )
})
class Model:
    @modal.build()
    def build(self):
        print("🛠️ Building container...")

        # Download model file
        os.makedirs(os.path.dirname(MODEL_FILE_PATH), exist_ok=True)
        with open(MODEL_FILE_PATH, "wb") as file:
            _ = file.write(requests.get(MODEL_URL).content)

        # Download VAE file
        os.makedirs(os.path.dirname(VAE_FILE_PATH), exist_ok=True)
        with open(VAE_FILE_PATH, "wb") as file:
            _ = file.write(requests.get(VAE_URL).content)

        # download_url = 'https://civitai.com/api/download/models/710901'
        # destination_file = f"/checkpoints/{MODEL_FILE_NAME}"

        # # Download the file with progress monitoring
        # def download_progress(count, block_size, total_size):
        #     percent = int(count * block_size * 100 / total_size)
        #     print(f"Downloading: {percent}%", end="\r")

        # urllib.request.urlretrieve(download_url, destination_file)
        # print("Download complete.")

        # print("☁️ r2 checkpoints folder:")
        subprocess.run(["ls", "/r2/models/checkpoints"])

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

        vae = AutoencoderKL.from_single_file(
            "https://huggingface.co/LyliaEngine/Pony_Diffusion_V6_XL/blob/main/sdxl_vae.safetensors",
            torch_dtype=torch.float16
        )
        pipe = StableDiffusionXLPipeline.from_single_file(
            "https://huggingface.co/LyliaEngine/Pony_Diffusion_V6_XL/blob/main/ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
            # vae=vae,
            safety_checker=None,
            torch_dtype=torch.float16,
        ).to("cuda")
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

        prompt = "score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, source_anime, a cat using a toaster"

        image = pipe(
            prompt=prompt,
            negative_prompt="",
            height=1024,
            width=1024,
            num_inference_steps=25,
            guidance_scale=5.0
        ).images[0]



        # vae = AutoencoderKL.from_single_file(VAE_FILE_PATH)
        # print("🔥 VAE IS:", vae)

        # pipeline = StableDiffusionXLPipeline.from_single_file(
        #     MODEL_FILE_PATH ,
        #     torch_dtype=torch.float16,
        #     useSafetensors=True,
        # )

        # pipeline.to("cuda")
        # print ("🔥 PIPELINE IS:", pipeline)

        # image = pipeline(
        #     vae = vae,
        #     prompt=prompt,
        #     negative_prompt="bad quality",
        #     num_inference_steps=20,
        #     denoising_end=0.8,
        #     width=512,
        #     height=512,
        # ).images[0]
        # print("🔥 IMAGE IS:", image)

        # Save the generated image
        output_path = "/r2/models/checkpoints/image.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)

        output_path = "/root/boogabooga/image.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)

        print(f"Image saved to: {output_path}")
        # byte_stream = io.BytesIO()
        # image.save(byte_stream, format="PNG")
        # img_str = base64.b64encode(byte_stream.getvalue()).decode()

        # print(f"Base64 string of the image: {img_str}")

        # image.save(byte_stream, format="JPEG",)
        # print("Image generated and saved as 'generated_image.png'.")


@app.local_entrypoint()
def main(prompt: str):
    Model().generate_image.remote(prompt)
