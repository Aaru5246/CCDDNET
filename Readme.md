# Enhanced Image Denoising via Color Component Decomposition and Bit-Plane Specific CNN Models

> 🚨 **This code is directly related to the manuscript currently submitted to _The Visual Computer_ journal.**

---


# Image Denoising Project

This repository contains CCDDNet, a deep learning-based model for bit-plane specific image denoising. It trains and denoises each RGB component separately using TensorFlow and OpenCV. All dependencies are listed in `requirements.txt`.

## 📂 Project Structure

```
├── train.ipynb       # Jupyter notebook for training the denoising model
├── testing.py        # Python script for testing the trained model
├── requirements.txt  # List of required packages

````

## 🔧 Installation

1. **Clone the repository**:

```bash
git clone https://github.com/Aaru5246/CCDDNET.git
cd your-repo-name
````

2. **Create and activate a virtual environment** (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate       # On Linux/macOS
venv\Scripts\activate          # On Windows
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 🔧 Training

Open the `train.ipynb` notebook and run the cells sequentially to train your denoising model.

### 🧪 Testing

Run the `test.py` script to test the model on new images:

```bash
python test.py
```

Modify `test.py` if needed to change input/output paths or model parameters.

## 📦 Requirements

* `numpy`
* `tensorflow`
* `opencv-python`

These are listed in `requirements.txt` for easy installation.