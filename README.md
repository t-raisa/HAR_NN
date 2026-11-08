# HAR_NN — Human Action Recognition with Neural Networks

## Project layout
- `data/` — put the dataset here (see `data/README.md`)
- `notebooks/01_preprocessing.ipynb` — step 1: load, explore and preprocess the images
- `notebooks/02_har_efficientnet.ipynb` — model 1: EfficientNetB0 transfer learning in Keras (76.4% validation accuracy)
- `notebooks/03_har_clip_vit.ipynb` — **model 2 (final)**: fine-tuned CLIP ViT-B/16 in PyTorch, with near-duplicate-aware train/validation/test split and 5-fold cross-validation (Part 7)

## Dataset
[Human Action Recognition (HAR) Dataset on Kaggle](https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset) — 15 actions, ~12,600 labeled images.
Run the notebooks on Kaggle and attach this dataset with **Add Input**; no upload needed.

On Kaggle, turn on **GPU** and **Internet** (needed to download the pretrained weights) before **Run All**.

## Results
| Model | Accuracy |
|---|---|
| Random guessing (15 actions) | 6.7% |
| EfficientNetB0, transfer learning (notebook 02) | 76.4% (validation) |
| CLIP ViT-B/16, fine-tuned (notebook 03) | **92.2% (test: 1,151 of 1,249 photos)** |

Notebook 03 trained in 13 minutes on a Kaggle T4 GPU. Its test set is held out from training and model selection,
and near-duplicate photos are kept on one side of the split. The hardest action is texting (F1 0.85),
mostly confused with other phone actions such as listening to music and calling.
