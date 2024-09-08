import subprocess
import modal
from pathlib import Path
import base64


image = (
    modal.Image.debian_slim(python_version="3.12.5")
    .apt_install("git")
    .pip_install("comfy-cli==1.1.6")
    # use comfy-cli to install the ComfyUI repo and its dependencies
    .run_commands("comfy --skip-prompt install --nvidia")
    # download all models and custom nodes required in your workflow
    .run_commands(
        "comfy --skip-prompt model download --url https://civitai.com/api/download/models/290640 --relative-path models/checkpoints"
    )
    .run_commands(
        "cd /root/comfy/ComfyUI/custom_nodes && git clone https://github.com/Chaoses-Ib/ComfyScript.git",
        "cd /root/comfy/ComfyUI/custom_nodes/ComfyScript && python -m pip install -e '.[default]'",
    )
)

app = modal.App("pony_diffusion_2")

# # Optional: serve the UI
# @app.function(
#     allow_concurrent_inputs=10,
#     concurrency_limit=1,
#     container_idle_timeout=30,
#     timeout=1800,
#     gpu="T4",
# )
# @modal.web_server(8000, startup_timeout=60)
# def ui():
#     _web_server = subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)

@app.cls(gpu="T4", container_idle_timeout=120, image=image)
class Model:
    # @modal.build()
    # def build(self):
    #     print("🛠️ Building container...")

    @modal.enter()
    def enter(self):
        print("✅ Entering container...")

        from comfy_script.runtime.real import load
        load()

        # subprocess.run("comfy manager disable-gui")

    # @modal.exit()
    # def exit(self):
    #     print("🧨 Exiting container...")

    @modal.method()
    def generate_image(self, prompt:str):
        print("🎨 Generating image...")
        from comfy_script.runtime.real import Workflow
        from comfy_script.runtime.real.nodes import CheckpointLoaderSimple, CLIPTextEncode, EmptyLatentImage, KSampler, VAEDecode, SaveImage, CivitAICheckpointLoader

        output_dir = "/root/comfy/ComfyUI/output"
        file_prefix = "generated_image"


        with Workflow():
            # model, clip, vae = CivitAICheckpointLoader('https://civitai.com/models/101055?modelVersionId=128078')
            model, clip, vae = CheckpointLoaderSimple("ponyDiffusionV6XL_v6StartWithThisOne.safetensors")
            conditioning = CLIPTextEncode(prompt, clip)
            conditioning2 = CLIPTextEncode('text, watermark', clip)
            latent = EmptyLatentImage(512, 1024, 1)
            latent = KSampler(model, 156680208300286, 20, 8, 'euler', 'normal', conditioning, conditioning2, latent, 1)
            image = VAEDecode(latent, vae)
            result = SaveImage(image, file_prefix)
            print("result", result)

        # returns the image as bytes
        for file in Path(output_dir).iterdir():
            if file.name.startswith(file_prefix):
                print(file)
                with open(file, "rb") as image_file:
                       print(base64.b64encode(image_file.read()))
            #     return file.read_bytes()

@app.local_entrypoint()
def main(prompt: str):
    Model().generate_image.remote(prompt)
