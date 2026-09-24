# PK-ZA-VGG16-GCN-Fossa: Prostate Cancer MRI Analysis

## Overview

This repository presents a multimodal deep-learning pipeline for **prostate MRI analysis and prostate cancer characterization**. The framework combines image preprocessing, Zernike-moment-based feature extraction, VGG16 deep features, graph convolutional learning, Fossa-inspired Grad-CAM visualization, and PI-RADS/Gleason-score interpretation.

The overall pipeline is designed to process **NIfTI (`.nii`) prostate MRI images and corresponding segmentation masks** and generate visual explanations of potentially relevant image regions.

### Main Pipeline

```text
Prostate MRI + Segmentation Mask
              │
              ▼
     Kernel Density Normalization
              │
              ▼
   Coherence Diffusion Filtering
              │
              ▼
       Contrast Enhancement
              │
              ├───────────────┐
              ▼               ▼
     Zernike Moments       MRI Slice
              │               │
              │               ▼
              │           VGG16
              │               │
              │               ▼
              │        Deep Image Features
              │               │
              └───────┬───────┘
                      ▼
             Graph Construction
                      │
                      ▼
          Graph Convolutional Network
                      │
                      ▼
              Attention Pooling
                      │
                      ▼
                Classification
                      │
                      ▼
        Fossa-Optimized YOLOv8
                      │
                      ▼
            Explainable Heatmap
                      │
                      ▼
        PI-RADS / Gleason Visualization
```

---

## Features

* NIfTI-based prostate MRI processing.
* Robust **Kernel Density Normalization (KDN)** using median and interquartile range.
* **Coherence Diffusion Filtering (CDF)** for image denoising.
* Intensity normalization for contrast enhancement.
* **Zernike moment** extraction using `mahotas`.
* Pre-trained **VGG16** feature extraction.
* Construction of an image-feature similarity graph.
* Graph Convolutional Network (GCN) for feature learning.
* Attention-based feature pooling.
* Explainability through gradient-based heatmap generation.
* Fossa-inspired optimization procedure for visual interpretation.
* YOLOv8-based feature/activation analysis.
* Visualization of MRI images, segmentation masks, and heatmaps.
* PI-RADS and corresponding Gleason Grade Group visualization.

---

## Dataset Structure

The code expects the dataset to be organized as follows:

```text
Dataset/
└── ProstateX/
    ├── images/
    │   ├── patient_001.nii
    │   ├── patient_002.nii
    │   ├── patient_003.nii
    │   └── ...
    │
    └── masks/
        ├── patient_001.nii
        ├── patient_002.nii
        ├── patient_003.nii
        └── ...
```

The default paths used by the implementation are:

```python
images_folder = r".\Dataset\ProstateX\images"
masks_folder = r".\Dataset\ProstateX\masks"
```

Each MRI image should have a corresponding segmentation mask.

---

## Input Data

The implementation processes:

* Prostate MRI volumes in NIfTI format.
* Corresponding prostate segmentation masks.
* A selected axial slice from the MRI volume.

The current implementation uses:

```python
image_data[:, :, 35]
```

as the working slice.

If a different slice is required, modify the slice index accordingly.

---

## Methodology

### 1. Kernel Density Normalization

The MRI volume is normalized using robust statistics:

```text
Normalized Image = (Image - Median) / IQR
```

where:

* `Median` is the median image intensity.
* `IQR` is the interquartile range.

This reduces the influence of extreme intensity values.

---

### 2. Coherence Diffusion Filtering

The normalized MRI data is passed through a total-variation-based diffusion filtering operation:

```python
denoise_tv_chambolle(
    image_data,
    weight=0.1,
    max_num_iter=10
)
```

The filtering stage aims to suppress noise while retaining important structural information.

---

### 3. Intensity Normalization

The filtered image is subsequently normalized to the `[0, 1]` range:

```text
I_normalized = (I - I_min) / (I_max - I_min)
```

This produces an enhanced image suitable for subsequent feature processing.

---

### 4. Zernike Moment Extraction

Zernike moments are extracted from the processed MRI slice using `mahotas`.

The image is resized to a circular region and Zernike moments are calculated using:

```python
radius = 21
degree = 8
```

These moments provide shape and spatial-structure descriptors.

---

### 5. VGG16 Feature Extraction

A pre-trained VGG16 network with ImageNet weights is used for deep feature extraction.

The MRI slice is:

1. Resized to `224 × 224`.
2. Converted from grayscale to three channels.
3. Passed through VGG16.
4. Converted into a feature representation.

The implementation uses:

```python
VGG16(
    weights="imagenet",
    include_top=False
)
```

and extracts convolutional features before the classification layers.

---

### 6. Graph Construction

The extracted image features are used to calculate cosine similarity.

```python
cosine_similarity(features)
```

The resulting similarity matrix is converted into a NetworkX graph:

```python
G = nx.from_numpy_array(image_similarity)
```

The graph represents relationships between image-feature representations.

---

### 7. Graph Convolutional Network

A custom graph convolution layer is implemented using TensorFlow/Keras.

The graph operation follows the general form:

```text
H' = AHW + b
```

where:

* `A` represents the adjacency/similarity matrix.
* `H` represents node features.
* `W` represents trainable weights.
* `b` represents bias.

The GCN uses:

```text
Graph Convolution
        ↓
Attention Pooling
        ↓
Dense 128
        ↓
Dense 64
        ↓
Sigmoid Output
```

---

### 8. Attention Pooling

An attention mechanism is applied to the graph-convolution output to obtain a compact representation.

The resulting representation is passed through fully connected layers for binary classification.

---

### 9. Fossa-Optimized Explainability

The framework generates gradient-based activation maps from a selected YOLOv8 convolutional layer.

The implementation uses:

```python
last_conv_layer_name = 'block14_sepconv2_act'
```

The gradients with respect to the selected feature map are used to generate a heatmap.

The heatmap is then overlaid onto the original MRI slice to identify regions contributing to the model representation.

---

### 10. PI-RADS and Gleason Visualization

The final visualization displays:

* Original MRI image.
* Segmentation/featured image.
* Fossa-optimized heatmap.
* PI-RADS score.
* Gleason score.
* Gleason Grade Group.

The implementation currently maps PI-RADS categories to example Gleason interpretations through:

```python
predict_Gleason_Score()
```

**Important:** PI-RADS is an imaging assessment system and should not be treated as a deterministic mapping to Gleason score. Histopathological Gleason grading requires appropriate clinical/pathological ground truth.

---

## Installation

Create a Python environment and install the required packages:

```bash
pip install tensorflow
pip install numpy
pip install scipy
pip install scikit-image
pip install nibabel
pip install mahotas
pip install matplotlib
pip install networkx
pip install scikit-learn
```

For the YOLOv8 model, ensure that the required trained Keras model is available:

```text
yolov8n-640.h5
```

The implementation also expects:

```text
LIME.pkl
```

to be available in the working directory.

---

## Required Files

Before execution, ensure the following resources are available:

```text
Dataset/
└── ProstateX/
    ├── images/
    └── masks/

yolov8n-640.h5
LIME.pkl
```

---

## Running the Code

From the repository root:

```bash
python main.py
```

or execute the notebook/script through Jupyter Notebook, JupyterLab, Spyder, or another Python IDE.

---

## Output

The implementation generates visualizations for the processed prostate MRI data.

### Preprocessing Visualization

```text
Original Image | Preprocessed Image
```

### Zernike Visualization

```text
Zernike Moment Magnitudes
```

### Feature/Mask Visualization

```text
Original Image | Featured Image
```

### Explainability Visualization

```text
Original Image | Featured Image | Heatmap
```

### Final Interpretation

```text
PI-RADS
Gleason Score
Grade Group
```

---

## Example Output Workflow

```text
Patient #1

┌──────────────────┬──────────────────┬──────────────────┐
│ Original MRI     │ Featured/Mask    │ Heatmap          │
│                  │                  │                  │
│      MRI         │  Segmentation    │ Activation Map   │
└──────────────────┴──────────────────┴──────────────────┘

PI-RADS: X
Gleason: X
Grade Group: X
```

---

## Project Structure

A recommended repository structure is:

```text
PK-ZA-VGG16-GCN-Fossa/
│
├── Dataset/
│   └── ProstateX/
│       ├── images/
│       └── masks/
│
├── main.py
├── yolov8n-640.h5
├── LIME.pkl
├── requirements.txt
├── README.md
└── results/
    ├── preprocessing/
    ├── zernike/
    ├── features/
    └── heatmaps/
```

---

## Dependencies

| Package            | Purpose                              |
| ------------------ | ------------------------------------ |
| TensorFlow / Keras | Deep learning and model construction |
| VGG16              | Deep feature extraction              |
| NumPy              | Numerical computation                |
| SciPy              | Image/volume processing              |
| scikit-image       | Resizing and denoising               |
| NiBabel            | NIfTI MRI loading                    |
| Mahotas            | Zernike moment extraction            |
| NetworkX           | Graph construction                   |
| scikit-learn       | Cosine similarity                    |
| Matplotlib         | Visualization                        |
| Pickle             | Loading stored prediction data       |

---

## Important Implementation Notes

### NIfTI Slice Selection

The current implementation uses slice index `35`:

```python
image_data[:, :, 35]
```

This assumes that the selected slice exists for the supplied volumes.

For datasets with different volume dimensions, an adaptive slice-selection strategy should be used.

### Mask Alignment

If the MRI and mask volumes have different dimensions, the mask is resized using nearest-neighbor interpolation:

```python
zoom(mask_data, resize_factors, order=0)
```

Nearest-neighbor interpolation is appropriate for categorical segmentation masks because it avoids introducing intermediate label values.

### Model Compatibility

The selected convolutional layer must exist in the loaded model:

```python
block14_sepconv2_act
```

If a different model architecture is used, the corresponding convolutional layer should be identified and supplied.

---

## Reproducibility and Model Development

For research use, the following should be explicitly defined:

* Patient-level train/validation/test splitting.
* Ground-truth labels.
* Training and validation protocol.
* Model checkpointing.
* Evaluation metrics.
* Hyperparameter configuration.
* Hardware/software versions.
* Dataset preprocessing protocol.

The repository should use actual clinical/pathological labels rather than generated labels for model evaluation.

---

## Recommended Evaluation Metrics

For binary prostate cancer classification, the following metrics can be reported:

```text
Accuracy
Precision
Recall / Sensitivity
Specificity
F1-Score
ROC-AUC
PR-AUC
MCC
```

For segmentation experiments:

```text
Dice Similarity Coefficient
IoU
Sensitivity
Specificity
Hausdorff Distance
```

For explainability analysis, qualitative heatmap visualization can be complemented with quantitative localization or overlap measures when appropriate ground-truth regions are available.
