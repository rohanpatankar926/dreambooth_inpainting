import os
from loguru import logger
import sys
import torch
import subprocess
from pathlib import Path
from contextlib import contextmanager

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HOME = Path.cwd()
GROUNDING_DINO_DIR = HOME / "GroundingDINO"
GROUNDING_DINO_CONFIG_PATH = GROUNDING_DINO_DIR / "groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = HOME / "weights/groundingdino_swint_ogc.pth"
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = HOME / "weights/sam_vit_h_4b8939.pth"
OUTPUT_DIR = "/notebooks/stable-diffusion-inpainting-furniture-sofa"

@contextmanager
def cd(directory):
    """Context manager to change the current working directory."""
    cwd = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(cwd)


def configure_segmentation():
    try:
        logger.info(f"HOME: {HOME}")
        with cd(HOME):
            logger.info("cloning https://github.com/IDEA-Research/GroundingDINO.git")
            subprocess.run("git clone https://github.com/IDEA-Research/GroundingDINO.git", check=True, shell=True)
            logger.info(f"Getting inside {GROUNDING_DINO_DIR}")
            with cd(GROUNDING_DINO_DIR):
                logger.info("git checking out to 57535c5a79791cb76e36fdb64975271354f10251")
                subprocess.run("git checkout -q 57535c5a79791cb76e36fdb64975271354f10251", check=True, shell=True)
                logger.info("installing supervision==0.6.0 ../")
                subprocess.run("pip install -q -e .", check=True, shell=True)
                subprocess.run("pip uninstall -y supervision", check=True, shell=True)
                subprocess.run("pip install -q supervision==0.6.0", check=True, shell=True)
                import supervision as sv
                logger.info(sv.__version__)
            logger.info("Roboflow installing")
            subprocess.run("pip install -q roboflow", check=True, shell=True)
            logger.info("making weight dir")
            (HOME / "weights").mkdir(parents=True, exist_ok=True)
            with cd(HOME / "weights"):
                logger.info("Downloading Pretrained layers")
                subprocess.run("wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth", check=True, shell=True)
                subprocess.run("wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", check=True, shell=True)
            subprocess.run(f"{sys.executable} -m pip install 'git+https://github.com/facebookresearch/segment-anything.git'", check=True, shell=True)
    except Exception as e:
        raise e
    

def configure_dreambooth_model():
    subprocess.run("pip install -qq git+https://github.com/ShivamShrirao/diffusers",check=True,shell=True)
    subprocess.run("wget -q https://raw.githubusercontent.com/huggingface/diffusers/main/examples/research_projects/dreambooth_inpaint/train_dreambooth_inpaint.py",check=True,shell=True)
    subprocess.run("pip install -qq git+https://github.com/ShivamShrirao/diffusers",check=True,shell=True)
    subprocess.run("pip install -q -U --pre triton",check=True,shell=True)
    subprocess.run("pip install -q accelerate transformers ftfy bitsandbytes==0.35.0 gradio natsort safetensors xformers",check=True,shell=True)
    subprocess.run("pip install -qq accelerate==0.16.0 tensorboard transformers ftfy gradio",check=True,shell=True)
    subprocess.run("python -c 'from accelerate import Accelerator; accelerator = Accelerator()'",check=True,shell=True)
    subprocess.run("pip install -qq accelerate tensorboard transformers ftfy gradio",check=True,shell=True)
    subprocess.run("mkdir -p ~/.huggingface",check=True,shell=True)
    HUGGINGFACE_TOKEN = "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" #@param {type:"string"}
    subprocess.run(f'''echo -n "{HUGGINGFACE_TOKEN}" > ~/.huggingface/token''',check=True,shell=True)
    subprocess.run("",check=True,shell=True)
    subprocess.run("",check=True,shell=True)
