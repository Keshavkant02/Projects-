# Intracranial Hemorrhage Segmentation using U-Net (Prototype)

## Overview

This project implements a prototype deep learning model based on the U-Net architecture to segment intracranial hemorrhage (ICH) regions from brain CT scan slices. It demonstrates a complete workflow from data acquisition and preprocessing to model training and evaluation using Python, TensorFlow/Keras, and common data science libraries.

This was developed as a learning exercise and to demonstrate practical skills in applying deep learning to medical imaging challenges, particularly addressing data loading and preprocessing hurdles encountered with real-world datasets. It serves as a portfolio piece showcasing foundational abilities relevant to roles involving medical image analysis and deep learning.

## Dataset

*   **Source:** [Computed Tomography Images for Intracranial Hemorrhage Detection and Segmentation](https://www.kaggle.com/datasets/vbookshelf/computed-tomography-ct-images) on Kaggle.
*   **Original Authors:** Murtadha D. Hssayeni, M.S., Muayad S. Croock, Ph.D., Aymen Al-Ani, Ph.D., Hassan Falah Al-khafaji, M.D. and Zakaria A. Yahya, M.D. (Please refer to dataset page for full citation and license details).
*   **Format:** The dataset contains brain CT slices saved as individual JPG files (grayscale, different window settings available). Corresponding binary segmentation masks highlighting hemorrhage regions are also provided as JPG files (black background, white hemorrhage region) with the naming convention `[slice_number]_HGE_Seg.jpg`.
*   **Details:** Includes 82 CT scans (~2800 slices total), with 318 slices containing annotated hemorrhage masks.

## Methodology / Workflow

The project follows these key steps, implemented in the accompanying Colab notebook (`.ipynb`):

1.  **Setup:** Installation and import of necessary Python libraries (TensorFlow, Keras, NumPy, Matplotlib, Scikit-image, Scikit-learn, Kaggle API).
2.  **Kaggle API Configuration:** Setting up API credentials (`kaggle.json`) for direct dataset download.
3.  **Data Acquisition:** Downloading and unzipping the dataset from Kaggle.
4.  **Data Loading & Preprocessing:**
    *   Iterating through patient folders and `brain` subfolders.
    *   Identifying corresponding image and mask files (using the `_HGE_Seg.jpg` suffix).
    *   Creating blank masks for slices without hemorrhage annotations.
    *   Reading JPG images and masks.
    *   Resizing images and masks to a uniform size (e.g., 128x128) using `skimage.transform.resize` (nearest neighbor interpolation for masks).
    *   Normalizing image pixel values to the [0, 1] range.
    *   Thresholding resized masks to ensure binary (0/1) output.
    *   Adding a channel dimension for compatibility with Keras.
    *   Splitting the data into training and validation sets using `train_test_split`.
5.  **Model Definition:** Implementing a standard U-Net architecture using the Keras functional API (`Input`, `Conv2D`, `MaxPooling2D`, `concatenate`, `UpSampling2D`, `Dropout`, `sigmoid` output).
6.  **Model Compilation:** Configuring the model for training using the Adam optimizer and Dice Loss (`1 - Dice Coefficient`). Metrics tracked: Dice Coefficient and binary accuracy.
7.  **Training:** Fitting the model (`model.fit`) to the training data for a specified number of epochs, using the validation set to monitor performance. **GPU acceleration is essential for this step.**
8.  **Evaluation:**
    *   Plotting training and validation loss/metric curves over epochs.
    *   Generating predictions on the validation set.
    *   Visualizing side-by-side comparisons of input images, ground truth masks, and model-predicted masks.

## Tools and Libraries

*   Python 3.x
*   TensorFlow / Keras
*   NumPy
*   Matplotlib
*   Scikit-image
*   Scikit-learn
*   Kaggle API
*   Google Colab (for environment and GPU access)

## Results (Example - Replace with your actual findings!)

The prototype model was trained for **[Number]** epochs (e.g., 10 epochs).

*   **Training Performance:** Achieved a final training Dice Coefficient of **[Value]** and validation Dice Coefficient of **[Value]**. *(Obtain these values from the final epoch output of Cell 10 or the plots in Cell 11)*.
*   **Learning Curves:**
    *(Insert screenshot of your Loss + Dice Coefficient plots from Cell 11 here. Upload the image to your GitHub repo first.)*
    ```
    ![Learning Curves](path/to/your/learning_curves.png)
    ```
    *(Comment briefly, e.g., "The plots show the loss generally decreasing and the Dice score increasing for both training and validation sets over the [Number] epochs. There is potential for further improvement with longer training, and the gap between training and validation suggests minimal overfitting at this stage.")*
*   **Sample Predictions:** Visual comparison demonstrates the model's capability to identify potential hemorrhage regions, although precision may vary.
    *(Insert one or two example screenshots from Cell 12 showing Image | True Mask | Predicted Mask. Upload images first.)*
    ```
    ![Sample Prediction 1](path/to/your/prediction_example_1.png)
    ![Sample Prediction 2](path/to/your/prediction_example_2.png)
    ```

## Challenges & Learnings

This project involved significant debugging, providing valuable learning experiences:

*   **Dataset Structure:** The initial file path assumed after unzipping was incorrect due to an extra parent directory created by the Kaggle download/unzip process. This required inspecting the file system and correcting the `DATA_ROOT` path in the loading script.
*   **Mask Filename:** The script initially failed to find mask files because the assumed filename (`[num]HGE_Seg.jpg`) was missing an underscore present in the actual files (`[num]_HGE_Seg.jpg`). Manual inspection using the file browser and correction of the filename pattern were necessary.
*   **Mask Loading & Thresholding:** This was the most challenging part. Ensuring the mask files (read as 0-255 `uint8`) were correctly converted to binary (0.0-1.0 `float32`) masks *after* resizing required iterative debugging. Adding print statements to check raw, resized, and processed mask values helped pinpoint that the thresholding logic needed adjustment (`> 128`) to reliably capture the white hemorrhage pixels after potential floating-point variations from resizing.
*   **Colab Environment:** Remembering to re-upload `kaggle.json` and re-run API setup (Cell 3) after kernel restarts (especially when switching to GPU) is crucial for interacting with the Kaggle API.

## How to Run

1.  Open the `.ipynb` notebook file in Google Colab.
2.  Ensure you have a `kaggle.json` API key file from your Kaggle account.
3.  Change the runtime type to use a **GPU** (`Runtime` -> `Change runtime type` -> `GPU`). The kernel will restart.
4.  Upload your `kaggle.json` file using the file browser (folder icon on the left).
5.  Run the cells sequentially from top to bottom.
    *   Cell 3 configures the Kaggle API.
    *   Cell 4 downloads and unzips the data (may take time).
    *   Cell 10 trains the model (requires GPU).

## Future Work & Improvements

*   **Train Longer:** Increase the number of epochs significantly (e.g., 50+).
*   **Data Augmentation:** Implement image augmentation (rotation, flips, brightness, etc.) to improve model robustness and handle the limited data size.
*   **Hyperparameter Tuning:** Experiment with learning rate, batch size, optimizer, and potentially U-Net architecture variations.
*   **Loss Functions:** Explore combined Dice + BCE Loss or Focal Loss.
*   **Input Size:** Test with larger image dimensions (e.g., 256x256) if resources permit.
*   **Patient-Level Split:** Implement a more robust validation split based on patient IDs.
*   **Explore 3D U-Net:** Adapt the pipeline for true volumetric segmentation.
