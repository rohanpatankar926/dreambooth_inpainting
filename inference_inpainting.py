from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image


def inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "/content/stable-diffusion-inpainting-furniture-sofa"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to(device)


    prompt = "a modern living room with a traditional sofa"

    image = Image.open("/content/istockphoto-968785508-612x612 (2).jpg")
    mask_image = Image.open("/content/mask.png")
    guidance_scale=8.5
    num_samples = 8
    generator = torch.Generator(device="cuda").manual_seed(0) # change the seed to get different results

    images = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=num_samples,
    ).images

    def image_grid(imgs, rows, cols):
        assert len(imgs) == rows*cols

        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols*w, rows*h))
        grid_w, grid_h = grid.size
        
        for i, img in enumerate(imgs):
            grid.paste(img, box=(i%cols*w, i//cols*h))
        return grid
    images.insert(0, image)
    image_grid(images, 1, num_samples + 1)


if __name__ == "__main__":
    inference()