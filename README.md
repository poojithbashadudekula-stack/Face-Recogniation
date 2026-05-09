# Face Recognition Project

This project implements a face recognition pipeline using Python and scikit-learn.
It extracts eigenfaces using PCA, applies LDA (Fisherfaces), and trains an MLP classifier to recognize faces.

## Files

- `face.py` - Main script that loads the dataset, trains the model, evaluates accuracy, and displays results.
- `live_sales_data.csv` - Additional data file (not used by `face.py`).
- `alarm.py/`, `lives.py` - Other project files in the workspace.

## Technologies

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- scikit-learn

## Setup

1. Install Python 3 if needed.
2. Install required packages:

```bash
pip install numpy matplotlib opencv-python scikit-learn
```

## Running the Project

Run the main script from the project directory:

```bash
python face.py
```

The script attempts to load the Labeled Faces in the Wild (LFW) dataset automatically using `sklearn.datasets.fetch_lfw_people`.

## Notes

- The script requires an internet connection the first time it downloads the LFW dataset.
- If internet is unavailable, you can provide a local dataset folder named `dataset/faces/` with subfolders for each person.
- Each subfolder should contain face image files for that person.

## Expected Behavior

- The script applies PCA to compute eigenfaces.
- It then reduces dimensionality using LDA.
- Finally, it trains an MLP classifier and prints the recognition accuracy.
- It also displays galleries of eigenfaces and test predictions.

## Submission Link

Repository link:

`https://github.com/poojithbashadudekula-stack/Face-Recogniation`

## Contact

For issues or improvements, edit `face.py` or update the README with additional instructions.
