# HAR_NN — Human Action Recognition with Neural Networks

## Project layout
- `data/` — put the dataset here (see `data/README.md`)
- `notebooks/01_preprocessing.ipynb` — step 1: load, explore and preprocess the images
- `notebooks/02_har_efficientnet.ipynb` — model 1: EfficientNetB0 transfer learning in Keras (76.4% validation accuracy)
- `notebooks/03_har_clip_vit.ipynb` — **model 2 (final)**: fine-tuned CLIP ViT-B/16 in PyTorch, with near-duplicate-aware train/validation/test split

## Dataset
[Human Action Recognition (HAR) Dataset on Kaggle](https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset) — 15 actions, ~12,600 labeled images.
Run the notebooks on Kaggle and attach this dataset with **Add Input**; no upload needed.

On Kaggle, turn on **GPU** and **Internet** (needed to download the pretrained EfficientNet weights) before **Run All**.
