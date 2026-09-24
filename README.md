# Kidney Image Analysis — FYP Project

AI-based system that classifies a kidney CT image as **Normal** or **Abnormal**
(Cyst/Tumor/Stone merged), shows a Grad-CAM heatmap of the region that drove
the decision, and generates a downloadable PDF report — built to match your
approved FYP proposal's methodology flowchart.

## Project structure

```
kidney_project/
├── organize_dataset.py       # Run locally: merges Cyst+Tumor+Stone into "Abnormal"
├── notebooks/
│   └── train_model.ipynb     # Run on Google Colab (needs GPU) — trains the model
└── app/
    ├── app.py                # Flask web app
    ├── gradcam.py            # Grad-CAM heatmap generation
    ├── report.py             # PDF report generation
    ├── requirements.txt
    ├── model/                # Put kidney_model.h5 here after training
    ├── templates/
    │   ├── index.html
    │   └── result.html
    └── static/
        ├── style.css
        └── uploads/          # Uploaded images + generated heatmaps land here
```

## Step-by-step: from raw dataset to a running app

### 1. Organize your dataset (on your own PC)
You already have the Kaggle CT Kidney dataset extracted, with folders
`Normal/`, `Cyst/`, `Tumor/`, `Stone/`. Run:

```bash
python organize_dataset.py --source "path/to/CT KIDNEY DATASET Normal, CYST, TUMOR and STONE" --dest "path/to/kidney_dataset"
```

This creates `kidney_dataset/Normal/` and `kidney_dataset/Abnormal/`.

### 2. Train the model on Google Colab
1. Zip the `kidney_dataset` folder and upload it to your Google Drive as `kidney_dataset.zip`.
2. Open `notebooks/train_model.ipynb` in Google Colab.
3. Go to **Runtime → Change runtime type → GPU**.
4. Run every cell top to bottom. Training takes roughly 20–40 minutes total
   (15 epochs frozen + 10 epochs fine-tuning) on a free Colab GPU.
5. The last cell saves `kidney_model.h5` to your Google Drive.

### 3. Set up the Flask app
1. Download `kidney_model.h5` from your Drive and place it in `app/model/`.
2. Install dependencies:
   ```bash
   cd app
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app.py
   ```
4. Open `http://127.0.0.1:5000` in your browser, upload a kidney CT image,
   and you'll get a Normal/Abnormal prediction with a heatmap and a PDF report.

## Important note on the Grad-CAM layer name

`app.py` uses `LAST_CONV_LAYER = "Conv_1"`, which is the correct last
convolutional layer name for MobileNetV2 in Keras. If you change the base
model (e.g. to ResNet50 or EfficientNet), run `model.summary()` in the
notebook, find the actual last conv layer's name, and update that constant
in `app.py` to match.

## What still needs your input for the final report

- Screenshot the training curves and evaluation metrics (accuracy, precision,
  recall, F1, confusion matrix, ROC-AUC) from the notebook — these go
  straight into your "Model testing and evaluation" deliverable (Week 8–9).
- Take screenshots of the working app (upload page, result page, PDF report)
  for your "Web/app interface development" deliverable (Week 10–11).
- Write up your literature review comparisons against your actual results
  in the final report.

## Disclaimer

This is an academic research tool, not a certified medical device. It should
never be used for real clinical decisions — only for demonstrating the
AI/ML pipeline as part of the FYP.
