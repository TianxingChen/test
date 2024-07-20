from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch
import cv2
import skimage
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers.image_utils import ImageFeatureExtractionMixin

class OWL_VIT:
    def __init__(self, device=None) -> None:
        self.score_threshold = 0.05
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

    def set_score_threshold(self, score_threshold):
        self.score_threshold = score_threshold
        
    # type: 0->over_threshold, top K 
    def get_box(self, image, text_queries, type=0, transform=True) -> dict: 
        orig_image_shape = image.shape

        processor = self.processor
        device = self.device
        model = self.model

        inputs = processor(text=text_queries, images=image, return_tensors="pt").to(device)
        # Print input names and shapes
        # for key, val in inputs.items():
        #     print(f"{key}: {val.shape}")

        # Set model in evaluation mode
        model = model.to(device)
        model.eval()

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)

        # for k, val in outputs.items():
        #     if k not in {"text_model_output", "vision_model_output"}:
        #         print(f"{k}: shape of {val.shape}")
        # print("\nText model outputs")
        # for k, val in outputs.text_model_output.items():
        #     print(f"{k}: shape of {val.shape}")

        # print("\nVision model outputs")
        # for k, val in outputs.vision_model_output.items():
        #     print(f"{k}: shape of {val.shape}") 

        mixin = ImageFeatureExtractionMixin()
        # Load example image
        image_size = model.config.vision_config.image_size
        image = mixin.resize(image, image_size)
        input_image = np.asarray(image).astype(np.float32) / 255.0

        # Threshold to eliminate low probability predictions
        score_threshold = self.score_threshold

        # Get prediction logits
        logits = torch.max(outputs["logits"][0], dim=-1)
        scores = torch.sigmoid(logits.values).cpu().detach().numpy()

        # Get prediction labels and boundary boxes
        labels = logits.indices.cpu().detach().numpy()
        boxes = outputs["pred_boxes"][0].cpu().detach().numpy()


        res_labels = []
        res_boxes = []
        res_scores = []

        if type == 0:
            for score, box, label in zip(scores, boxes, labels):
                if score < score_threshold:
                    continue
                res_labels.append(label)
                res_boxes.append(box)
                res_scores.append(score)
        else: # top K
            K = min(type, scores.shape[0])
            idxs = self.get_top_k_indices(scores, K)
            
            for i in range(K):
                res_labels.append(labels[idxs[i]])
                res_boxes.append(boxes[idxs[i]])
                res_scores.append(scores[idxs[i]])

        if transform:
            for i in range(len(res_boxes)): # transform [0, 1] -> image coordinate
                cx, cy, w, h = res_boxes[i]
                cx *= orig_image_shape[1]
                cy *= orig_image_shape[0]
                w *= orig_image_shape[1]
                h *= orig_image_shape[0]
                cx -= w/2
                cy -= h/2
                res_boxes[i] = np.array([cx, cy, w, h])

        return res_boxes, res_labels, res_scores

    def get_top_k_indices(self, arr, k):
        indices = np.argpartition(arr, -k)[-k:]
        return indices

        
    def plot_predictions(self, input_image, text_queries, res):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(input_image, extent=(0, 1, 1, 0))
        ax.set_axis_off()
        scores = res['score']
        boxes = res['box']
        labels = res['label']
        
        for score, box, label in zip(scores, boxes, labels):
            cx, cy, w, h = box
            ax.plot([cx-w/2, cx+w/2, cx+w/2, cx-w/2, cx-w/2],
                    [cy-h/2, cy-h/2, cy+h/2, cy+h/2, cy-h/2], "r")
            ax.text(
                cx - w / 2,
                cy + h / 2 + 0.015,
                f"{text_queries[label]}: {score:1.5f}",
                ha="left",
                va="top",
                color="red",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "red",
                    "boxstyle": "square,pad=.3"
                })
        plt.show()
