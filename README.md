# Scalable Multimodal Encoder for Mental Health Disorder Detection

### NULab - Northeastern University, Boston, USA  
**Graduate Research Assistant**  
*January 2023 - July 2023*  

This project is part of research conducted under the supervision of Dr. Silvio Amir at NULab, Northeastern University. The focus of the research was on developing a scalable multimodal encoder to detect various mental health disorder types, utilizing proprietary social media data. The dataset used contains approximately 2 million data points from social media posts, making it one of the largest multimodal datasets applied to mental health detection tasks.

**Note:** Certain files have been omitted from this repository due to a Non-Disclosure Agreement (NDA). These files include proprietary dataset handling and specific training/inference codes.

## Project Overview

The primary objective of this project was to build a scalable and efficient model that could detect different types of mental health disorders from multimodal data (text and images). The encoder was designed to integrate both textual and visual information to capture context more accurately and enhance predictive performance.

### Key Contributions
- **Multimodal Encoder**: A state-of-the-art encoder capable of processing both text and images to classify mental health disorder types.
- **Pre-training Strategies**: Implemented multiple pre-training strategies, including DINO (self-supervised vision pre-training) and Masked Language Modeling (MLM) for text. These techniques contributed to an attention localization score of **87.6%**, a significant achievement in the domain of mental health prediction.
- **Scalability**: The model was trained on a large-scale dataset (~2 million data points), ensuring it can handle real-world applications with significant data volumes.

## Repository Structure

This repository contains the following directories and files:

- **config/** : Contains configuration files for various aspects of the project, including model training, inference settings, and hyperparameters.
- **dataset/** : Data loading utilities for text and image data (excluding proprietary dataset-specific code, showcasing Hateful Memes dataset which is similar to the proprietary data).
- **examples/** : Scripts demonstrating the application of models on open datasets and their respective training pipelines.
- **models/** : Core model architectures, including fusion strategies for text and image data.
- **preprocessing/** : Scripts for preprocessing text and image data, including embedding generation and fusion.
- **supervised/** : Training scripts for supervised learning tasks (generalized for public datasets).
- **loss_functions/** : Custom loss functions for optimizing the multimodal models.

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/mldlwizard/LITART-Literature-to-Artistic-Representations-with-NLP-and-Computer-Vision.git
cd LITART-Literature-to-Artistic-Representations-with-NLP-and-Computer-Vision
pip install -r requirements.txt
```
## Usage
Due to the proprietary nature of the dataset used in this project, certain files and training/inference steps are omitted. However, the general pipeline for multimodal learning is provided, allowing the model to be applied to publicly available datasets.

To run inference with a sample model configuration:
```bash
python Run_This_Inference.py --config config/inference/late_fusion.yaml
```
## Pre-training Strategies
The following pre-training strategies were employed during model development:

- **DINO (Self-supervised Vision Pre-training)/**: DINO was used to learn high-quality image representations without labeled data. This technique allowed the model to capture detailed image features crucial for multimodal fusion.
- **Masked Language Modeling (MLM)/**: A pre-training task where certain words are masked in a sentence, and the model learns to predict them. This improves the model’s understanding of text-based context.
These pre-training strategies contributed to an impressive attention localization score of 87.6%.

## Disclaimer
Files related to the proprietary dataset and specific training/inference processes have been omitted due to an NDA signed before research. This repository contains all the publicly shareable components of the project.
