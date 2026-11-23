# Notebook 04: models, settings and results

## Results (Kaggle run, test set of 1,249 photos)

Accuracy is taken at each model's best epoch (highest validation accuracy). Early stopping ends a run after
3 epochs without a better validation accuracy.

| Model | Type | Parameters | Epochs run | Best epoch | Train | Validation | Test |
|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | small CNN | 4.0M | 23 | 20 | 88.4% | 73.3% | 74.5% |
| ResNet-50 | classic CNN | 23.5M | 15 | 12* | 78.5%* | 72.2%* | 74.9%* |
| ConvNeXt-Tiny | modern CNN | 27.8M | 15 | 12 | 97.9% | 87.1% | 85.5% |
| **CLIP ViT-B/16** | Vision Transformer | 85.8M | **25** | **24** | 99.9% | **90.9%** | **90.0%** |

\* ResNet-50's log is missing the lines for epochs 11–12, so its best-epoch numbers are unknown; the numbers shown are from its last epoch (15), which were within about half a point of the best.

Optional check runs (CLIP ViT-B/16, one change each; the run stopped before they finished):

| Run | Change | What happened |
|---|---|---|
| Weight decay 1e-3 | 4e-4 → 1e-3 | Best validation 87.3% at epoch 1, stopped at epoch 4. Inconclusive: it stopped because epoch 4 only tied epoch 1. |
| No dropout | 0.05 → 0 | Interrupted at epoch 5 (validation 86.9%, test 88.4%). Not finished. |
| Smaller ViT-S | 86M → 22M weights | Not reached. |

**Takeaways**
- CLIP ViT-B/16 is the best model: **90.0% test**, and it used **24 of the 25 epochs**, so the teacher's 20–25 epochs
  were needed. Early stopping never triggered for it.
- ConvNeXt-Tiny is the best CNN (85.5%). EfficientNet-B0 and ResNet-50 stay around 74–75%.
- All models overfit to some degree: training accuracy reaches 88–100%, validation stays lower. Dropout, weight decay,
  augmentation and early stopping keep it in check.
- ResNet-50 stopped with only 78.5% training accuracy: the small learning rate (2e-5), chosen so CLIP keeps its
  pretrained knowledge, is slow for this CNN. With one shared learning rate for all models, that is the trade-off.

## Why each model

All four are **pretrained** networks (transfer learning). Each was trained once before on a large dataset, and we
fine-tune it on our 15 actions. We use a pretrained model because 12,600 photos are too few to train a deep network from zero.

| Model | Why it is in the comparison |
|---|---|
| EfficientNet-B0 | Our first model, now trained with the same settings as the others. Small and fast; the baseline. |
| ResNet-50 | The classic CNN most papers compare against. Introduced skip connections. |
| ConvNeXt-Tiny | A modern CNN, redesigned with ideas from transformers; pretrained on ImageNet-22k (14M photos). |
| CLIP ViT-B/16 | A Vision Transformer: self-attention lets distant parts of the photo interact (hand, phone, ear). The strongest pretrained image model we could fine-tune on Kaggle's GPU. |

### "We don't have captions, so why CLIP?"
We don't use captions. CLIP is two networks trained together: an **image encoder** (a Vision Transformer) and a text encoder.
They were trained on 400M internet photos with captions, which taught the image encoder to recognise what people are doing.

We keep **only the image encoder** and throw the text encoder away. We add our own 15-neuron output layer and fine-tune
it on **our photos and our labels**, with normal cross-entropy loss, exactly like the ImageNet models.

It's the same idea as ResNet-50: that model learned from ImageNet's 1,000 object labels, and we reuse it for our 15 action
labels. Captions were only how CLIP learned its starting weights. Captions describe actions ("a man talking on the phone"),
while ImageNet labels name objects ("telephone"), so CLIP's starting point suits action recognition better. The results show it:
90.0% for CLIP vs. 85.5% for the best ImageNet model.

## The settings (the only five)

| Setting | Value | Why |
|---|---|---|
| `DROPOUT` | 0.05 | Teacher's value. Randomly switches off 5% of the features before the output layer during training, so the model can't rely on single features. Small, because augmentation and weight decay also fight overfitting. |
| `EPOCHS` | 25 | Teacher's range (20–25). It's the maximum; early stopping can end sooner. CLIP used 24. |
| `PATIENCE` | 3 | Teacher's value. Stop when validation accuracy hasn't improved for 3 epochs, and keep the best epoch's weights. Uses the validation set, never the test set. |
| `LEARNING_RATE` | 2e-5 | Small, because the pretrained weights are already good; big steps destroy them. At 5e-5 CLIP's validation accuracy fell from 79.9% to 72.7% after one epoch. The new output layer starts random, so it uses 10× (2e-4). |
| `WEIGHT_DECAY` | 4e-4 | Each step shrinks the weights slightly toward zero (AdamW), which discourages extreme weights and reduces overfitting. 1e-3 was tested as a comparison. |

## Fixed parts (same for every model, not tuned)

| Part | Value | What it is |
|---|---|---|
| Data split | 80 / 10 / 10% (10,094 / 1,257 / 1,249) | Train / validation / test, same as notebook 03. Near-duplicate photos are kept on one side. |
| Image size | 224 × 224 | The input size these pretrained models expect. |
| Normalization | each model's own mean and std | Inputs on the same scale the model saw in pretraining. |
| Augmentation | random crop 40–100%, flip, rotate ±10°, brightness/contrast ±20% | Training photos only; same as notebook 03. |
| Batch size | 32 | Photos per weight update; fits in the T4 GPU's memory. |
| Optimizer | AdamW | Standard optimizer for fine-tuning; adaptive step per weight, with weight decay. |
| Schedule | 1 warm-up epoch, then cosine decay | The rate starts tiny so the random output layer can't damage the pretrained layers, then decreases smoothly. |
| Loss | cross-entropy, label smoothing 0.1 | The standard classification loss; smoothing avoids 100%-confident targets for ambiguous photos. |
| Gradient clipping | 1.0 | Caps very large updates. |
| Mixed precision | FP16 | About 2× faster on the GPU, no effect on accuracy. |
| Output | dropout → dense layer (15 neurons) → softmax | Softmax turns 15 scores into probabilities that add up to 1. |
