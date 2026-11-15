"""
SmartPark AI - Scikit-Learn Slot Patch Classifier
Academic baseline: Feature Extraction (HSV Histogram, Edge Density, Texture Variance)
combined with Scikit-learn Classifier (SVM / Random Forest) for parking slot occupancy.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import glob
import random
import cv2
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from config import PATCH_CLASSIFIER_PATH, SEGMENTED_PKLOT_DIR


class PatchClassifier:
    """
    Extracts computer vision features from individual parking slot patches
    and classifies them as Vacant (0) or Occupied (1).
    """

    def __init__(self, model_path: Union[str, Path] = PATCH_CLASSIFIER_PATH):
        self.model_path = Path(model_path)
        self.pipeline: Optional[Pipeline] = None
        self._try_load()

    def _try_load(self) -> bool:
        """Attempt to load pretrained model pipeline from disk."""
        if self.model_path.exists():
            try:
                self.pipeline = joblib.load(str(self.model_path))
                return True
            except Exception:
                self.pipeline = None
        return False

    @staticmethod
    def extract_features(patch_bgr: np.ndarray) -> np.ndarray:
        """
        Extract compact CV feature vector from an image patch:
        - HSV Color Histogram (24 dims: 8 Hue, 8 Saturation, 8 Value)
        - Canny Edge Density (1 dim)
        - Gray Intensity Standard Deviation (1 dim)
        - Laplacian Gradient Variance (1 dim)
        Total: 27 feature dimensions.
        """
        # Resize to standard patch dimension
        resized = cv2.resize(patch_bgr, (64, 64))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # 1. Color histograms
        h_hist = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()

        # Normalize histograms
        h_hist /= (np.sum(h_hist) + 1e-6)
        s_hist /= (np.sum(s_hist) + 1e-6)
        v_hist /= (np.sum(v_hist) + 1e-6)

        # 2. Canny edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / (64.0 * 64.0)

        # 3. Gray standard deviation (contrast)
        gray_std = float(np.std(gray)) / 255.0

        # 4. Laplacian variance (texture roughness / sharpness)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 1000.0

        features = np.concatenate([
            h_hist,
            s_hist,
            v_hist,
            [edge_density, gray_std, laplacian_var]
        ])
        return features.astype(np.float32)

    def train_on_segmented_pklot(self,
                                 max_samples_per_class: int = 500,
                                 segmented_dir: Union[str, Path] = SEGMENTED_PKLOT_DIR) -> Dict[str, float]:
        """
        Train a Scikit-Learn SVM classifier on PKLot segmented crops.
        """
        segmented_path = Path(segmented_dir)
        empty_files = glob.glob(str(segmented_path / "**" / "Empty" / "*.jpg"), recursive=True)
        occupied_files = glob.glob(str(segmented_path / "**" / "Occupied" / "*.jpg"), recursive=True)

        if not empty_files or not occupied_files:
            raise RuntimeError(f"No segmented images found in {segmented_dir}")

        random.seed(42)
        sample_empty = random.sample(empty_files, min(len(empty_files), max_samples_per_class))
        sample_occ = random.sample(occupied_files, min(len(occupied_files), max_samples_per_class))

        X = []
        y = []

        for p in sample_empty:
            img = cv2.imread(p)
            if img is not None:
                X.append(self.extract_features(img))
                y.append(0)

        for p in sample_occ:
            img = cv2.imread(p)
            if img is not None:
                X.append(self.extract_features(img))
                y.append(1)

        X = np.array(X)
        y = np.array(y)

        # Train/Test split
        idx = np.random.permutation(len(X))
        split = int(0.8 * len(X))
        train_idx, test_idx = idx[:split], idx[split:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Pipeline with Scaler + LogisticRegression / LinearSVC
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0))
        ])

        self.pipeline.fit(X_train, y_train)

        # Evaluate
        preds = self.pipeline.predict(X_test)
        acc = float(accuracy_score(y_test, preds))

        # Save model
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, str(self.model_path))

        return {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "test_accuracy": round(acc * 100.0, 2)
        }

    def predict_patch(self, patch_bgr: np.ndarray) -> Tuple[bool, float]:
        """
        Predict occupancy for a single parking slot patch.
        Returns (is_occupied, confidence).
        """
        if self.pipeline is None:
            if not self._try_load():
                # Fallback to simple edge heuristic if model not trained yet
                feat = self.extract_features(patch_bgr)
                edge_dense = feat[-3]
                is_occ = edge_dense > 0.045
                return is_occ, 0.75

        feat = self.extract_features(patch_bgr).reshape(1, -1)
        pred = int(self.pipeline.predict(feat)[0])

        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(feat)[0]
            conf = float(probs[pred])
        else:
            conf = 0.85

        return (pred == 1), conf
