# Plant Disease Classification with Attention‑Guided Explainability
This repository contains the code and results for a mini research project on improving Grad‑CAM faithfulness in plant disease classification using attention‑guided retraining. The work is part of a deep learning course at The Hong Kong Polytechnic University.

## Project Overview
Deep learning models can achieve high accuracy on plant disease datasets, but they often rely on spurious background cues rather than actual disease symptoms. This project:

1. Fine‑tunes a ResNet‑18 model on a subset of the PlantVillage dataset (Apple, Corn, Grape) to perform binary healthy/diseased classification.
2. Manually annotates 160 training images and 40 test images with disease region masks.
3. Introduces a weighted binary cross‑entropy attention loss during retraining to force the model’s feature maps to align with the masks.
4. Quantitatively evaluates explainability using stability (noise robustness), agreement (Grad‑CAM vs. Simple Gradients), and alignment (heatmap vs. ground truth mask).

The retrained model shows significantly better alignment and stability while maintaining near‑perfect classification accuracy (>99%).

## Dataset
The PlantVillage dataset is downloaded automatically via kagglehub (https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset). The code selects three plants (Apple, Corn, Grape) and splits them into healthy/diseased classes. The training/validation/test split uses 400/100/400 images per class. A separate set of 160 training images (20 per disease) and 40 test images (5 per disease) is manually annotated with disease masks (in COCO JSON format). The masks are stored in annotations/.../masks/.

## Code and Data Availability
- The full code is in this repository.
- The manually annotated masks are included in annotations/ in three splits (i.e. 'train', 'val' and 'test'). Each split contains 'images' and 'masks' which store the images selected and the corresponding annotations.
- Raw PlantVillage images are automatically downloaded by the notebooks.
