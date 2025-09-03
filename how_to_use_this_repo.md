# Urban Sound Classification -- Quick Start Guide

## 📥 1. Clone the repository

``` bash
git clone https://github.com/your-username/Urban_sound_classification.git
cd Urban_sound_classification
```

------------------------------------------------------------------------

## 🎵 2. Download the dataset

-   Get the **UrbanSound8K dataset** from Kaggle:\
    👉 [Kaggle
    Link](https://www.kaggle.com/datasets/chrisfilo/urbansound8k)

-   Place it in a folder named **`archive/`** inside the repo.\

-   The structure should look like this:

```{=html}
<!-- -->
```
    archive/
    ├─ fold1/
    ├─ fold2/
    ├─ fold3/
    │   ...
    ├─ fold10/
    └─ UrbanSound8K.csv   # metadata file, can be found in the same kaggle data.

Each `foldX/` contains `.wav` files from the dataset.

------------------------------------------------------------------------

## ⚙️ 3. Create and activate environment

``` bash
# Create venv
python -m venv env_sound_classification

# Activate (Linux/Mac)
source env_sound_classification/bin/activate

# Activate (Windows)
env_sound_classification\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🚀 4. Run the pipeline

### Step 1 -- Extract features

``` bash
python src/extract/extract_data.py
```

### Step 2 -- Train models

``` bash
python src/training/train.py --archive_df_dir archive_dataframes --target class
```

------------------------------------------------------------------------

## 🔹 5. Optional: Generate Training & Evaluation Plots

If you want to regenerate the visualizations (training comparisons, KNN
analysis, evaluation metrics), we provide a helper script.

1.  Make sure you have already trained models and that
    `static/metrics.csv` exists.
2.  From the **root of the repo**, run:

``` bash
cd D:\Github_work\Urban_sound_classification
python static/training_results_plots/make_training_plots.py
```

3.  The plots will be saved automatically into:

```{=html}
<!-- -->
```
    static/training_results_plots/
        ├─ bar_best5_worst5.png
        ├─ line_knn_neighbors_f1.png
        ├─ bar_summary_best3_worst2.png
        ├─ eval_best_model_bars.png
        ├─ eval_precision_recall_scatter.png
        └─ eval_f1_hist.png

These images are referenced inside the main `README.md` file to
illustrate model training and evaluation results.


------------------------------------------------------------------------

## 🌐 6. Run the app

After training is done, start the web app:

``` bash
python app.py
```

------------------------------------------------------------------------

✅ That's it! You now have a working **Urban Sound Classification
pipeline + app**.
