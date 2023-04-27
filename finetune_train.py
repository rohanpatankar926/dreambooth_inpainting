import os
from typing import List
import cv2
import numpy as np
import numpy as np
import cv2
from config import *
from groundingdino.util.inference import Model
import supervision as sv
from segment_anything import SamPredictor
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

annotated_fram=[]
annotated_img=[]
class GroundingDino:
    def __init__(self,image_path,bedroom_items = ['Chairs',"Books",'Bed', 'Pillows', 'Blankets', 'Bedsheets', 'Mattresses', 'Nightstands', 'Wardrobes', 'Dressers', 'Mirrors', 'Table lamps', 'Curtains', 'Windows', 'Doors', 'Rugs', 'Alarm clock', 'Ceiling fans', 'Air conditioners', 'Heaters', 'TV', 'Clothes hangers', 'Clothes racks']):
        first=[classes.lower() for classes in bedroom_items[0:7]]
        second=[classes.lower() for classes in bedroom_items[7:14]]
        third=[classes.lower() for classes in bedroom_items[14:21]]
        fourth=[classes.lower() for classes in bedroom_items[21:28]]
        self.classes=[{"bed":first,"chair":second,"wall":third,"windows":fourth}]
        self.image_path=image_path
        HOME=Path.cwd()
        os.chdir(f"{HOME}/GroundingDINO")

    def enhance_class_name(self,class_names: List[str]) -> List[str]:
        return [
            f"all {class_name}s"
            for class_name
            in class_names
        ]

    def bbox_detector(self,box_threshold=0.35,text_threshold=0.25):
        for category in self.classes[0]:
            #Bounding box detection code from here
            image = cv2.imread(self.image_path)
            detections = grounding_dino_model.predict_with_classes(
                image=image,
                classes=self.enhance_class_name(class_names=self.classes[0][category]),
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
            data=[]
            zipped_data=zip(detections.class_id,detections.confidence,detections.xyxy)
            for i in zipped_data:
                if i[0] is not None:
                    data.append(i)
            detections.class_id=np.array([cid[0] for cid in data],dtype=object)
            detections.confidence=np.array([conf[1] for conf in data])
            detections.xyxy=np.array([xy[2] for xy in data])
            box_annotator = sv.BoxAnnotator()
            labels = [f"{self.classes[0][category][class_id]} {confidence:0.2f}" for _,_,confidence,class_id, _ in detections]
            annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
            
            #segmentation starts from here
            sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
            sam_predictor = SamPredictor(sam)
            detections.mask = self.segment(
                sam_predictor=sam_predictor,
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy)
            box_annotator = sv.BoxAnnotator()
            mask_annotator = sv.MaskAnnotator()
            labels = [
                f"{self.classes[0][category][class_id]} {confidence:0.2f}" 
                for _,_, confidence, class_id, _ 
                in detections]
            annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
            annotated_fram.append(annotated_frame)
            annotated_img.append(annotated_image)
        return annotated_fram,annotated_img
    
    def plot_bbox(self,annotated_frame):
        for frames in annotated_frame:
          sv.plot_image(frames, (30, 30))
        
    def plot_seg(self,segmented_frame):
      for segs in segmented_frame:
        sv.plot_image(segs, (30, 30))

    def segment(self,sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)




def train_dreambooth_model():
        try:
            subprocess.run('''!python3 train_dreambooth_inpaint.py \
                --pretrained_model_name_or_path="runwayml/stable-diffusion-inpainting" \
                --instance_data_dir="/storage/data_bed" \
                --output_dir="/notebooks/stable-diffusion-inpainting-furniture-sofa" \
                --resolution=512 \
                --train_batch_size=1 \
                --gradient_accumulation_steps=2 \
                --learning_rate=1e-4 \
                --lr_scheduler="constant" \
                --lr_warmup_steps=0 \
                --use_8bit_adam \
                --num_class_images=11000 \
                --max_train_steps=500 \
                --instance_prompt="photo of bed" \
                --class_prompt="a photo of furniture" \
                --train_text_encoder \
                --gradient_checkpointing''',check=True,shell=True)
        except subprocess.CalledProcessError as e:
            print(e)
            return e
