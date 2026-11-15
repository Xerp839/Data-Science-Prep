# Mind-Scope — Brain Tumor Detection CNN (your college project, explained)

> Everything here comes from your actual code (`notebook90ac12de34.ipynb`, `home.py`, `project.py`), so the numbers are real. Read this once on Day 2 and re-read the "30-second pitch" + "weaknesses" on Thursday morning.

---

## Should you bring it up? ✅ Yes, but smartly

- The JD says *"Train, fine-tune, and evaluate ML and DL models"* and *"Strong fundamentals in ML and DL"*. Your resume has **zero** ML training work. This project fills that hole.
- **Don't open with it.** Lead with Quantal. Bring it in when they ask about ML/DL, CNNs, overfitting, metrics or model training.
- Consider adding it to your printed resume as one line under "Projects (Academic)".
- **Be honest about where the code came from.** The model-building and preprocessing code closely follows a popular public Kaggle/GitHub brain-tumor notebook (same `crop_brain_contour`, `build_model` and helper functions). That's normal for a student project. If someone asks "did you write this from scratch?", say: *"We started from a reference implementation and adapted it. We trained it on our augmented dataset, evaluated it, and built the Streamlit app and the PDF report on top."* Honesty here protects you. If they've seen that notebook and you claim it as 100% original, it hurts you.
- Small correction: your About page says "CT-scan", but the dataset (`yes/` and `no/` folders, files like `Y256.JPG`, `N20.JPG`) is the well-known **brain MRI** dataset. **Say "MRI images".**

---

## 🎤 30-second pitch (memorize)

> "In college I built Mind-Scope, a brain tumor detection system. It's a binary classifier on brain MRI images: tumor or no tumor. I preprocessed the scans with OpenCV: grayscale, Gaussian blur, thresholding, erosion and dilation to find the brain contour, then cropped to just the brain and resized to 240 by 240. The model is a compact CNN in Keras: a 7×7 convolution with 32 filters, batch normalization, ReLU, two max-pool layers and a sigmoid output, only about 11 thousand parameters. Trained with Adam and binary cross-entropy, it got about 91% test accuracy and a 0.915 F1 score. I deployed it as a Streamlit app that also generates a downloadable PDF report. Looking back, it overfit noticeably, and I'd use transfer learning and focus on recall, since missing a tumor is the costly error."

---

## 1. The Problem
- **Task:** Binary image classification. Input: brain MRI image → Output: **Tumor (1)** or **No Tumor (0)**.
- **Data:** Kaggle brain MRI dataset (`yes` / `no` folders). The original set is small (a couple hundred images), so it was **augmented** (rotations, flips etc.) to **2,065 images**.
  - 1,085 tumor (52.5%) / 980 no tumor (47.5%). **Roughly balanced**, so accuracy is a fair metric here.
- **Split:** 70% train / 15% validation / 15% test → **1,445 / 310 / 310** images.

---

## 2. The Preprocessing Pipeline (the part you remembered as "polishing, Gaussian, black and white")

Function: `crop_brain_contour(image)`. **Goal: cut away the black background and keep only the brain**, so the model focuses on brain tissue and not on empty space or scanner borders.

```
Original MRI (BGR color)
   │
   ▼  Step 1: Grayscale ............ cv2.cvtColor(BGR → GRAY)
   ▼  Step 2: Gaussian Blur ........ cv2.GaussianBlur(5×5)
   ▼  Step 3: Threshold ............ pixel > 45 → white(255), else black(0)
   ▼  Step 4: Erode ×2 ............. remove small white noise specks
   ▼  Step 5: Dilate ×2 ............ grow the main shape back to size
   ▼  Step 6: Find contours ........ take the LARGEST one = the brain
   ▼  Step 7: Extreme points ....... leftmost, rightmost, topmost, bottommost
   ▼  Step 8: Crop ................. cut the ORIGINAL image to that box
   ▼  Step 9: Resize ............... 240 × 240 (bicubic interpolation)
   ▼  Step 10: Normalize ........... pixel / 255 → values 0–1
   ▼
Model input: (240, 240, 3)
```

### What each step does and WHY

| Step | What | Why (say this) |
|---|---|---|
| **1. Grayscale** | 3 color channels → 1 intensity channel | MRI is basically grayscale anyway. Contour detection works on intensity, and it's simpler and faster. |
| **2. Gaussian Blur (5×5)** | Smooths the image with a bell-curve-weighted average of nearby pixels | **Reduces noise and tiny details** so the threshold step gives a clean shape instead of jagged speckles. |
| **3. Binary Threshold (45)** | Every pixel becomes pure black or pure white | Separates the **brain (bright)** from the **background (dark)**. This is the "black and white" step you remembered. |
| **4. Erosion (×2)** | Shrinks white regions | Removes small white noise blobs and thin artifacts (like text or scanner marks). |
| **5. Dilation (×2)** | Expands white regions | Restores the brain region to its original size after erosion. **Erode then dilate = "morphological opening"**, a classic noise-removal combo. |
| **6. Find largest contour** | Outlines all white shapes, keeps the biggest | The biggest shape is the brain. Everything else is noise. |
| **7–8. Extreme points + crop** | Finds the bounding edges of the brain and crops | Removes useless black background → **the model sees more brain and less nothing**. Also makes images consistent. |
| **9. Resize 240×240** | All images the same size | CNNs need a fixed input size. |
| **10. Normalize ÷255** | Pixels 0–255 → 0–1 | Small, consistent input values make gradient descent **train faster and more stably**. |

> Note: the grayscale/blur/threshold image is used **only to find where to crop**. The crop is applied to the **original color image**, which is why the model input still has 3 channels.

---

## 3. The CNN Architecture — Layer by Layer

```python
X_input = Input((240, 240, 3))
X = ZeroPadding2D((2, 2))(X_input)
X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
X = BatchNormalization(axis=3, name='bn0')(X)
X = Activation('relu')(X)
X = MaxPooling2D((4, 4), name='max_pool0')(X)
X = MaxPooling2D((4, 4), name='max_pool1')(X)
X = Flatten()(X)
X = Dense(1, activation='sigmoid', name='fc')(X)
```

| # | Layer | Output shape | Params | What it does (plain English) |
|---|---|---|---|---|
| 0 | **Input** | 240 × 240 × 3 | 0 | The preprocessed image |
| 1 | **ZeroPadding2D (2,2)** | 244 × 244 × 3 | 0 | Adds a 2-pixel border of zeros so the 7×7 filter can cover edge pixels properly and the image doesn't shrink too much |
| 2 | **Conv2D 32 filters, 7×7, stride 1** | 238 × 238 × 32 | **4,736** | 32 different 7×7 filters slide over the image, each learning to detect a pattern (edges, textures, bright blobs like tumors). Output = 32 "feature maps". |
| 3 | **BatchNormalization** | 238 × 238 × 32 | **128** | Normalizes each of the 32 channels (mean≈0, std≈1) → faster, more stable training |
| 4 | **ReLU** | 238 × 238 × 32 | 0 | Keeps positives, zeroes out negatives → adds non-linearity |
| 5 | **MaxPool 4×4** | 59 × 59 × 32 | 0 | Keeps the strongest value in each 4×4 block → 16× fewer values, keeps the important signals |
| 6 | **MaxPool 4×4** | 14 × 14 × 32 | 0 | Shrinks again → compact summary of features |
| 7 | **Flatten** | 6,272 | 0 | Unrolls 14×14×32 into one long vector |
| 8 | **Dense(1, sigmoid)** | 1 | **6,273** | One neuron → probability of tumor (0 to 1). **> 0.5 → "Tumor Detected"** |
| | **Total** | | **11,137** | (11,073 trainable + 64 non-trainable) |

### Be ready to calculate these (they love asking)
- **Conv output size:** `(N − F + 2P)/S + 1` → padding already applied: `(244 − 7)/1 + 1 = 238` ✅
- **Conv params:** `(7 × 7 × 3) weights × 32 filters + 32 biases = 4,704 + 32 = 4,736` ✅
- **BatchNorm params:** 4 per channel × 32 = 128 → **gamma & beta (trainable, 64)** + **moving mean & variance (non-trainable, 64)**. That's where the "64 non-trainable" comes from ✅
- **MaxPool:** 238 / 4 = 59 (floored), 59 / 4 = 14 ✅
- **Flatten:** 14 × 14 × 32 = 6,272 ✅
- **Dense params:** 6,272 weights + 1 bias = 6,273 ✅

---

## 4. Training Setup

| Setting | Value | Why |
|---|---|---|
| Optimizer | **Adam** | Adaptive learning rate, the default go-to |
| Loss | **Binary cross-entropy** | Binary classification with a sigmoid output |
| Metric | Accuracy (+ F1 computed after) | Classes are roughly balanced |
| Batch size | 32 | ~46 steps per epoch (1,445 / 32) |
| Epochs | Several runs (~10 + extra rounds of 3–5) | |
| **ModelCheckpoint** | Saves the model only when **val_accuracy improves** | Keeps the **best** model, not the last one → `cnn-parameters-improvement-09-0.92.keras` = epoch 09, val acc 0.92 |
| **TensorBoard** | Logs loss/accuracy curves | To visualize training |

## 5. Results

| Metric | Value |
|---|---|
| **Test accuracy** | **91.3%** |
| **Test F1** | **0.915** |
| Validation F1 | 0.915 |
| Best validation accuracy | ~0.92 |

## 6. Deployment (the Streamlit app)
- `app.py` → sidebar navigation with **Home** (detection) and **About** (tumor info) pages.
- `home.py`:
  1. User uploads a PNG/JPG.
  2. The image is resized to 240×240 and normalized to /255.
  3. `model.predict()` → probability → > 0.5 = "Tumor Detected".
  4. Generates a **PDF report** (ReportLab) with the timestamp, diagnosis, recommendation, the image and a disclaimer.
- The model is **loaded once at app start** (module level), not on every prediction. *Same principle as your Presidio fix at Quantal, and a nice link to make!*

---

## 7. ⚠️ Weaknesses & What You'd Improve (this makes you look senior)

Interviewers often ask *"what would you do differently?"* Here are real, specific answers:

1. **Overfitting.** Training accuracy reached ~99.9% while validation swung between ~71% and ~92% across epochs. The model memorized the training set.
   → *Fix:* Dropout, L2 regularization, **early stopping**, on-the-fly augmentation, and above all **transfer learning** (pretrained VGG16 / ResNet50 / EfficientNet with a fine-tuned top).

2. **Possible data leakage from augmentation.** Images were augmented *before* the train/val/test split, so augmented copies of the same original scan could land in both train and test. That can inflate the test score.
   → *Fix:* **Split first by original patient/image, then augment only the training set.**

3. **Train/serve preprocessing mismatch.** In training, images went through `crop_brain_contour` and were read with OpenCV (**BGR** channel order). In the Streamlit app, images are **not cropped** and are read with PIL (**RGB** order). So production inputs look different from training inputs.
   → *Fix:* Put one shared `preprocess()` function in both training and serving. *(This "training-serving skew" is a real MLOps concept, and naming it scores points.)*

4. **Wrong metric focus for a medical problem.** We optimized accuracy with a 0.5 threshold. In medical screening, **recall (sensitivity)** matters most, because a missed tumor is far worse than a false alarm.
   → *Fix:* Report the confusion matrix, recall and ROC-AUC, and **lower the decision threshold** to catch more tumors.

5. **Very shallow architecture.** Only one conv layer, then two aggressive 4×4 pools that throw away a lot of spatial detail.
   → *Fix:* Stack several smaller 3×3 conv blocks (conv→BN→ReLU→pool) or use a pretrained backbone.

6. **Small dataset, no external validation.** A couple hundred original images is tiny. The model may not generalize to other hospitals or scanners.
   → *Fix:* More data, cross-validation, testing on an independent dataset.

7. **Explainability.** Doctors need to know *why*.
   → *Fix:* **Grad-CAM** heatmaps showing which region of the scan drove the prediction.

---

## 8. Likely Questions & Answers

**Q: Why CNN and not a normal neural network?**
A: An image has spatial structure. CNNs use small filters that slide across the image, share weights, and detect local patterns like edges and textures. A dense network on 240×240×3 = 172,800 inputs would have far too many parameters and would ignore pixel neighborhoods.

**Q: Why Gaussian blur before thresholding?**
A: To smooth out noise. Without it, thresholding would produce speckles and broken edges, and contour detection would pick up junk.

**Q: Why erode then dilate?**
A: Erosion removes small white noise blobs. Dilation restores the main brain shape. Together it's "morphological opening", which cleans the mask before we find the contour.

**Q: Why sigmoid and not softmax?**
A: It's binary with a single output neuron. Sigmoid gives P(tumor). Softmax would need 2 output neurons. Both work, but sigmoid + binary cross-entropy is the standard for binary problems.

**Q: What does BatchNorm do?**
A: It normalizes activations per batch so each layer gets stable inputs. That speeds up training, allows higher learning rates, and adds slight regularization.

**Q: Why max pooling?**
A: It downsamples (fewer computations), keeps the strongest features, and gives some tolerance to small shifts in position.

**Q: How did you choose the best model?**
A: ModelCheckpoint saved the model whenever validation accuracy improved. The best was epoch 9 at ~0.92 val accuracy. I then evaluated it **once** on the held-out test set: 91.3% accuracy, 0.915 F1.

**Q: Was the data imbalanced?**
A: Only slightly, about 52.5% tumor and 47.5% no tumor, so accuracy is meaningful. But I'd still report recall because of the medical context.

**Q: How would you improve it today?**
A: Transfer learning with a pretrained ResNet/EfficientNet, split before augmenting, one shared preprocessing for train and serve, optimize for recall with threshold tuning, add Grad-CAM for explainability, and serve it via a Flask/FastAPI endpoint with monitoring instead of only Streamlit.

**Q: Precision or recall for this?**
A: Recall. A false negative means a patient with a tumor is told they're fine. A false positive just leads to a follow-up scan.
