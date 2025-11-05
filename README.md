# Indian Coin Classification using SVM and CNN

This project tackles the problem of automatic Indian coin classification. Given an image of an Indian coin, the goal is to identify its denomination (e.g., 1 Rupee, 5 Rupee, 50 Paisa). This is an important task for systems like automated vending machines, cash sorting, or digital numismatics.

This project explores and compares two main approaches:
1.  **Traditional Computer Vision:** Using Histogram of Oriented Gradients (HOG) as a feature descriptor, combined with Support Vector Machine (SVM) classifiers.
2.  **Deep Learning:** Employing Convolutional Neural Networks (CNNs), starting from a baseline model and progressing to a state-of-the-art transfer learning model (EfficientNetB0).

The results show that deep learning approaches, particularly the fine-tuned EfficientNetB0 model, significantly outperform the HOG+SVM method, achieving the highest validation accuracy.

---

## Dataset source

* **Dataset:** The project uses the **Indian Coins Image Dataset**.
* **Data Volume:** The dataset contains **6672 images** belonging to 7 different classes (denominations).
* **Preprocessing:**
    * All images were loaded and resized to **(128, 128) pixels**.
    * **For the HOG+SVM approach:** Images were converted to grayscale, and a Gaussian Blur filter was applied before feature extraction.
    * **For the CNN approaches:** Images were kept in RGB format (except for one grayscale baseline) and pixel values were normalized by rescaling (dividing by 255.0).
    * **Data Augmentation:** For the improved CNN and EfficientNet models, data augmentation was applied during training, including `RandomRotation`, `RandomZoom`, and `RandomFlip`.

---

## Methods

This project compares two main methodologies for solving the image classification problem.

### Approach 1: HOG Features + Support Vector Machine (SVM)

This is a traditional machine learning approach that first engineers features from the images and then uses a classifier.

1.  **Feature Extraction (HOG):**
    * Images are preprocessed (grayscale, blur).
    * **Histogram of Oriented Gradients (HOG)** features are extracted from each image. HOG is effective at capturing shape and texture. This process resulted in a feature vector of size 8100 for each image.
2.  **Feature Scaling:**
    * Before training, the HOG features are scaled using `StandardScaler` to normalize the data.
3.  **Classification (SVM):**
    * A **Support Vector Machine (SVM)** classifier is trained on the scaled HOG features.
    * **Alternative Kernels:** This approach was evaluated by comparing four different SVM kernels:
        * Radial Basis Function (RBF)
        * Linear
        * Polynomial (Poly)
        * Sigmoid

### Approach 2: Deep Learning (Convolutional Neural Networks)

This approach uses end-to-end deep learning models that learn features directly from the raw pixel data.

1.  **Baseline CNN (Grayscale & RGB):**
    * A simple CNN model (`model`) was first built and trained on grayscale images, which resulted in very low accuracy.
    * The same architecture (`model_rgb`) was then trained on **RGB (color) images**, which dramatically improved performance.

2.  **Improved CNN (with Regularization):**
    * A second CNN was built (`model_improved`) in an attempt to improve the RGB model, incorporating `Dropout` and `BatchNormalization`.
    * **Data Augmentation:** This model was trained on an augmented dataset (rotations, zooms, flips) to improve generalization.

3.  **Transfer Learning (EfficientNetB0):**
    * This approach uses a powerful, pre-trained model, **EfficientNetB0**, as a feature extractor.
    * **Base Model:** The convolutional base of EfficientNetB0 (pre-trained on ImageNet) was frozen.
    * **Custom Head:** A new classification head was added on top, consisting of `GlobalAveragePooling2D`, `BatchNormalization`, `Dense`, `Dropout`, and a final `Dense (softmax)` output layer.
    * **Fine-Tuning:** After initial training, the model was fine-tuned by unfreezing the top 25 layers of the base model and re-training with a very low learning rate (`1e-5`).

---

## Steps to run the code

1.  **Prerequisites:**
    * Ensure you have a Python environment with `tensorflow`, `scikit-learn`, `scikit-image`, `opencv-python`, `numpy`, and `matplotlib` installed.
    * `pip install scikit-image scikit-learn tensorflow opencv-python numpy matplotlib`

2.  **Dataset:**
    * Download the "Indian_Coins_Image_Dataset".
    * Place the dataset in your Google Drive at the following path: `/content/drive/MyDrive/Indian_Coins_Image_Dataset/`.
    * The dataset directory must contain sub-folders for each class (e.g., `1 Rupee Coin`, `10 Rupee Coin`, etc.).

3.  **Execution:**
    * Open the `coin_classification.ipynb` notebook in Google Colab.
    * Mount your Google Drive to allow Colab to access the dataset.
    * Run the notebook cells sequentially.

---

## Experiments/Results summary

This table summarizes the final accuracy achieved by each method as plotted in the project's final comparison graph. The accuracy for SVM and baseline/improved CNNs is the **Test Accuracy**, while the accuracy for EfficientNetB0 models is the final **Validation Accuracy**.

| Model | Accuracy |
| :--- | :--- |
| HOG + SVM (RBF Kernel) | 0.6682 |
| CNN (Grayscale) | 0.1220 |
| Regularized CNN (Grayscale) | 0.6585 |
| CNN (RGB) | 0.8064 |
| Improved CNN (RGB) | 0.7835 |
| EfficientNetB0 (Base Frozen) | 0.7414 |
| **EfficientNetB0 (Fine-Tuned)** | **0.8133** |

### HOG + SVM Model Comparison

This table shows the final test accuracy for each SVM kernel evaluated.

| SVM Kernel | Test Accuracy |
| :--- | :--- |
| **RBF** | **0.6682** |
| Sigmoid | 0.6232 |
| Linear | 0.6082 |
| Polynomial | 0.5768 |

### CNN Model Results

| Model | Input |  Accuracy |
| :--- | :--- | :--- |
| Baseline CNN | Grayscale | 0.1220 |
| Regularized CNN | Grayscale | 0.6585 |
| Baseline CNN | RGB | 0.8064 |
| Improved CNN | RGB + Augmentation | 0.7835 |
| EfficientNetB0 (Base Frozen) | RGB + Augmentation | 0.7414 |
| **EfficientNetB0 (Fine-Tuned)** | **RGB + Augmentation** | **0.8133** |

* **Grayscale vs. RGB:** The baseline CNN's performance starkly highlighted the importance of color, jumping from **12.2%** accuracy on grayscale to **80.6%** on RGB images.
* **Transfer Learning:** EfficientNetB0 (Base Frozen) achieved a validation accuracy of **74.1%**. The fine-tuned EfficientNetB0 model achieved the highest validation accuracy of **81.3%**.

### Visualizations

**1. Final Model Comparison (Bar Chart)**
This graph provides a direct comparison of the final accuracy scores from all models.
plots/compareall.png

**2. CNN Model Training History (Loss)**
This graph compares the training and validation loss for the CNN model.
plots/improvedEfficientNet.png

---

## Conclusion

This project successfully demonstrates and compares different methods for Indian coin classification. The key findings are:

1.  **Deep Learning is Superior:** CNN models significantly outperformed the traditional HOG+SVM approach. The best HOG+SVM model (RBF kernel) only achieved **66.8%** accuracy, while the best CNN model reached **81.3%**.
2.  **Color is a Critical Feature:** Using RGB images provided a massive performance boost for CNNs, with accuracy jumping from **12.2%** (grayscale) to **80.6%** (RGB).
3.  **Transfer Learning is Effective:** The fine-tuned EfficientNetB0 model, pre-trained on ImageNet, adapted well to this specific task and achieved the highest validation accuracy (**81.3%**), demonstrating the power of transfer learning.

---

## References

Dataset: https://data.mendeley.com/datasets/txn6vz28g9/2
