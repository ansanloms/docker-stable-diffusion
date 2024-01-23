import os
import math
import uuid
import diffusers
import torch
import datetime

prompt = """
<lora:nj5furry-v1:1>,
a photograph of an astronaut riding a horse
"""

negative_prompt = """
nsfw,
cropped,
(bad face, bad eyes, bad anatomy, bad hands, missing fingers: 1.6),
(worst quality, low quality, normal quality: 1.5),
tatoo,accessory,wearing,clothes,
color,
"""


def slim_prompt(prompt):
    rep = "".join(
        filter(
            lambda x: str.strip(x) != "" and str.strip(x)[:2] != "//",
            prompt.split("\n"),
        )
    )

    return "".join(
        map(
            lambda x: str.strip(x),
            rep.replace(",", ",\n").split("\n"),
        )
    )


print(slim_prompt(prompt))

checkpoint_path = os.path.join(
    os.path.dirname(__file__),
    "../models/indigoFurryMix_se01Vpred.safetensors"
)

original_config_file = os.path.join(
    os.path.dirname(__file__),
    "../models/indigoFurryMix_se01Vpred.yaml"
)

weight_name = os.path.join(
    os.path.dirname(__file__),
    "../models/nj5furry-v1.safetensors"
)

pipeline = diffusers.StableDiffusionPipeline.from_single_file(
    checkpoint_path,
    original_config_file=original_config_file,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    device_map="auto",
).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

pipeline.safety_checker = None
pipeline.requires_safety_checker = False

pipeline.enable_vae_slicing()
pipeline.enable_xformers_memory_efficient_attention()

pipeline.load_lora_weights(".", weight_name=weight_name)


def generate(pipeline, num_inference_steps):
    def progress(pipeline, step_index, timestep, callback_kwargs):
        if (math.floor((step_index / num_inference_steps) * 5) - math.floor(((step_index - 1) / num_inference_steps) * 5)) >= 1:
            with torch.no_grad():
                latents = 1 / 0.18215 * callback_kwargs["latents"]
                for index, image in enumerate(pipeline.numpy_to_pil(
                    (pipeline.vae.decode(latents).sample / 2 + 0.5)
                        .clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()
                )):
                    image.save(os.path.join(
                        os.path.dirname(__file__),
                        "../outputs/generate/",
                        f"step_{step_index}.png"
                    ))

        return callback_kwargs

    images = pipeline(
        prompt=slim_prompt(prompt),
        negative_prompt=slim_prompt(negative_prompt),
        width=960,
        height=1280,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=1,
        seed=-1,
        callback_on_step_end=progress,
    ).images

    for index, image in enumerate(images):
        image_filename = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4()}"
        image.save(os.path.join(
            os.path.dirname(__file__),
            "../outputs/",
            f"{image_filename}.png"
        ))


for num in range(1):
    generate(pipeline, 150)
