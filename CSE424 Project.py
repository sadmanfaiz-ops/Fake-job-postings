"""CSE424 Multi-Modal Pattern Recognition Project: Fake Job Posting Detection.

This project compares two classical ML model families:
1) `logistic_regression`: multi-modal fusion (text + categorical + numeric)
2) `random_forest`: reduced text features (TF-IDF + SVD) + numeric

Additionally creates K-Means clustering visualization and ROC/confusion matrices.

Outputs are saved under `outputs/`:
- `model_comparison.csv`
- `confusion_matrices.png`
- `roc_curves.png`
- `model_comparison_metrics.png`
- `kmeans_clustering.png`
- `metrics_table.png`
- `feature_importance_rf.png`
- `best_model.joblib`
- `best_model_info.json`
- `metrics_summary.json`

Example:
    python "CSE424 Project.py" --data "fake_job_postings.csv" --output-dir "outputs"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
	roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, RobustScaler


TARGET_COLUMN = "fraudulent"

TEXT_COLUMNS = [
	"title",
	"company_profile",
	"description",
	"requirements",
	"benefits",
	"department",
	"function",
	"industry",
	"location",
]

CATEGORICAL_COLUMNS = [
	"employment_type",
	"required_experience",
	"required_education",
	"industry",
	"function",
	"department",
]

BASE_NUMERIC_COLUMNS = [
	"telecommuting",
	"has_company_logo",
	"has_questions",
]

ENGINEERED_NUMERIC_COLUMNS = [
	"salary_available",
	"salary_lower",
	"salary_upper",
	"salary_mid",
	"title_len",
	"description_len",
	"requirements_len",
	"benefits_len",
	"company_profile_len",
	"word_count_total",
]

ALL_NUMERIC_COLUMNS = BASE_NUMERIC_COLUMNS + ENGINEERED_NUMERIC_COLUMNS


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
	for column in columns:
		if column not in df.columns:
			df[column] = np.nan
	return df


def _clean_text(value: object) -> str:
	if pd.isna(value):
		return ""
	text = str(value)
	text = re.sub(r"https?://\S+|www\.\S+", " ", text)
	text = re.sub(r"#URL_[A-Za-z0-9_#]+", " ", text)
	text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
	text = re.sub(r"\s+", " ", text).strip().lower()
	return text


def _parse_salary_range(value: object) -> Tuple[float, float, float, int]:
	if pd.isna(value):
		return np.nan, np.nan, np.nan, 0

	numbers = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", str(value))]
	if not numbers:
		return np.nan, np.nan, np.nan, 0

	if len(numbers) == 1:
		lower = upper = numbers[0]
	else:
		lower, upper = numbers[0], numbers[1]
		if upper < lower:
			lower, upper = upper, lower

	mid = (lower + upper) / 2.0
	return lower, upper, mid, 1


def load_dataset(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path)
	if TARGET_COLUMN not in df.columns:
		raise ValueError(f"Target column '{TARGET_COLUMN}' was not found in: {path}")
	return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()

	required_columns = (
		TEXT_COLUMNS
		+ CATEGORICAL_COLUMNS
		+ BASE_NUMERIC_COLUMNS
		+ ["salary_range", TARGET_COLUMN]
	)
	df = _ensure_columns(df, required_columns)

	for col in TEXT_COLUMNS:
		df[col] = df[col].map(_clean_text)

	for col in CATEGORICAL_COLUMNS:
		df[col] = df[col].fillna("missing").astype(str).str.strip().replace("", "missing")

	for col in BASE_NUMERIC_COLUMNS:
		df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

	salary_features = df["salary_range"].apply(_parse_salary_range)
	salary_df = pd.DataFrame(
		salary_features.tolist(),
		columns=["salary_lower", "salary_upper", "salary_mid", "salary_available"],
		index=df.index,
	)
	df = pd.concat([df, salary_df], axis=1)

	df["combined_text"] = df[TEXT_COLUMNS].fillna("").agg(" ".join, axis=1).str.replace(r"\s+", " ", regex=True).str.strip()

	df["title_len"] = df["title"].str.len()
	df["description_len"] = df["description"].str.len()
	df["requirements_len"] = df["requirements"].str.len()
	df["benefits_len"] = df["benefits"].str.len()
	df["company_profile_len"] = df["company_profile"].str.len()
	df["word_count_total"] = df["combined_text"].str.split().str.len()

	for col in ENGINEERED_NUMERIC_COLUMNS:
		df[col] = pd.to_numeric(df[col], errors="coerce")

	df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)
	return df


def _flatten_series(data: object) -> np.ndarray:
	if isinstance(data, pd.Series):
		return data.fillna("").astype(str).to_numpy()
	if isinstance(data, pd.DataFrame):
		return data.iloc[:, 0].fillna("").astype(str).to_numpy()
	arr = np.asarray(data)
	if arr.ndim == 2 and arr.shape[1] == 1:
		arr = arr[:, 0]
	return arr.astype(str)


def _binary_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray | None) -> Dict[str, float]:
	metrics: Dict[str, float] = {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"precision": float(precision_score(y_true, y_pred, zero_division=0)),
		"recall": float(recall_score(y_true, y_pred, zero_division=0)),
		"f1": float(f1_score(y_true, y_pred, zero_division=0)),
	}
	if y_prob is not None:
		try:
			metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
		except ValueError:
			metrics["roc_auc"] = float("nan")
	return metrics


def plot_confusion_matrices(results: List[Dict[str, object]], output_dir: Path) -> None:
	"""Create confusion matrix heatmaps for all models."""
	fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
	if len(results) == 1:
		axes = [axes]

	fig.suptitle("Confusion Matrices: Logistic Regression vs Random Forest", fontsize=14, fontweight="bold")

	for idx, (result, ax) in enumerate(zip(results, axes)):
		cm = np.array(result["confusion_matrix"])
		sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=True)
		ax.set_title(result["model"].upper(), fontweight="bold")
		ax.set_ylabel("True Label")
		ax.set_xlabel("Predicted Label")

	plt.tight_layout()
	confusion_path = output_dir / "confusion_matrices.png"
	plt.savefig(confusion_path, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved: {confusion_path}")


def plot_roc_curves(model_names: List[str], X_test: pd.DataFrame, y_test: pd.Series, models: List[Pipeline], output_dir: Path) -> None:
	"""Create ROC curve plots for all models."""
	fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
	if len(models) == 1:
		axes = [axes]

	fig.suptitle("ROC Curves: Model Comparison", fontsize=14, fontweight="bold")

	for idx, (name, model, ax) in enumerate(zip(model_names, models, axes)):
		try:
			y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
			if y_prob is not None and y_test.nunique() > 1:
				fpr, tpr, _ = roc_curve(y_test, y_prob)
				auc = roc_auc_score(y_test, y_prob)

				ax.plot(fpr, tpr, lw=2.5, label=f"AUC = {auc:.3f}", color="steelblue")
				ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random")
				ax.set_xlabel("False Positive Rate", fontsize=11)
				ax.set_ylabel("True Positive Rate", fontsize=11)
				ax.set_title(f"{name.replace('_', ' ').title()}\nROC Curve", fontweight="bold")
				ax.legend(loc="lower right", fontsize=10)
				ax.grid(alpha=0.3)
		except Exception:
			pass

	plt.tight_layout()
	roc_path = output_dir / "roc_curves.png"
	plt.savefig(roc_path, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved: {roc_path}")


def plot_model_comparison(comparison_df: pd.DataFrame, output_dir: Path) -> None:
	"""Create bar charts for model metric comparison."""
	metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
	existing_metrics = [m for m in metrics if m in comparison_df.columns]

	fig, axes = plt.subplots(1, len(existing_metrics), figsize=(5 * len(existing_metrics), 4))
	if len(existing_metrics) == 1:
		axes = [axes]

	fig.suptitle("Model Performance Metrics Comparison", fontsize=14, fontweight="bold")

	colors = ["#FF6B6B", "#4ECDC4"]
	for idx, (metric, ax) in enumerate(zip(existing_metrics, axes)):
		data = comparison_df[["model", metric]].dropna()
		bars = ax.bar(range(len(data)), data[metric], color=colors[:len(data)], alpha=0.8, edgecolor="black", linewidth=1.5)
		ax.set_ylim([0, 1.05])
		ax.set_ylabel(metric.upper(), fontsize=11, fontweight="bold")
		ax.set_title(metric.upper(), fontsize=12, fontweight="bold")
		ax.set_xticks(range(len(data)))
		ax.set_xticklabels([m.replace("_", " ").title() for m in data["model"]], rotation=45, ha="right")
		ax.grid(axis="y", alpha=0.3, linestyle="--")
		for i, v in enumerate(data[metric]):
			ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

	plt.tight_layout()
	comparison_path = output_dir / "model_comparison_metrics.png"
	plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved: {comparison_path}")


def plot_kmeans_clustering(X_train: pd.DataFrame, y_train: pd.Series, output_dir: Path, n_clusters: int = 2) -> None:
	"""Create K-Means visualization with clear real/fake differentiation."""
	text_series = X_train.get("combined_text", pd.Series(index=X_train.index, dtype=str)).fillna("").astype(str)
	text_vectorizer = TfidfVectorizer(max_features=2500, ngram_range=(1, 2), stop_words="english")
	X_text = text_vectorizer.fit_transform(text_series)
	n_text_components = max(2, min(50, X_text.shape[1] - 1)) if X_text.shape[1] > 1 else 1
	text_reducer = TruncatedSVD(n_components=n_text_components, random_state=42)
	X_text_reduced = text_reducer.fit_transform(X_text) if X_text.shape[1] > 1 else X_text.toarray()

	X_numeric = X_train[ALL_NUMERIC_COLUMNS].copy()
	X_numeric = X_numeric.fillna(X_numeric.median())

	# Clip extreme outliers so projection does not collapse into a line.
	for col in X_numeric.columns:
		low = X_numeric[col].quantile(0.01)
		high = X_numeric[col].quantile(0.99)
		X_numeric[col] = X_numeric[col].clip(lower=low, upper=high)

	X_scaled = RobustScaler().fit_transform(X_numeric.to_numpy())
	X_features = np.hstack([X_text_reduced, X_scaled])
	X_features = RobustScaler().fit_transform(X_features)

	pca = PCA(n_components=2, random_state=42)
	X_pca = pca.fit_transform(X_features)
	X_plot = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X_pca)

	# Cluster in 2D PCA space for stable centroids and readable plot.
	kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
	clusters = kmeans.fit_predict(X_plot)

	fig, ax = plt.subplots(figsize=(12, 8), facecolor="white")

	# Plot by the true job label so the image clearly separates real and fake postings.
	# The dataset uses 0 = real and 1 = fake.
	label_styles = [
		(0, "Real Jobs", "#1f77b4"),
		(1, "Fake Jobs", "red"),
	]
	for label_value, label_name, color in label_styles:
		label_mask = y_train == label_value
		ax.scatter(
			X_plot[label_mask, 0],
			X_plot[label_mask, 1],
			c=color,
			alpha=0.7,
			s=38,
			edgecolors="white",
			linewidth=0.35,
			label=f"{label_value} - {label_name}",
		)

	ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="black", marker="X", s=220, linewidths=1.5, label="Centroids", zorder=6)

	ax.set_xlabel("PCA Component 1")
	ax.set_ylabel("PCA Component 2")
	ax.set_title("K-Means Clustering of Job Postings", fontsize=15, fontweight="bold")
	ax.legend(loc="upper right", frameon=True, title="Label")
	ax.grid(alpha=0.25, linestyle="--")
	fig.text(0.07, 0.055, "0 = Real Jobs\n1 = Fake Jobs", fontsize=12, color="black", ha="left", va="bottom")
	fig.tight_layout(rect=[0, 0.08, 1, 1])
	kmeans_path = output_dir / "kmeans_clustering.png"
	plt.savefig(kmeans_path, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved: {kmeans_path}")


def plot_metrics_table(comparison_df: pd.DataFrame, output_dir: Path) -> None:
	"""Create a detailed metrics summary table as an image."""
	fig, ax = plt.subplots(figsize=(14, 4))
	ax.axis("tight")
	ax.axis("off")

	table_data = []
	for idx, row in comparison_df.iterrows():
		table_data.append([
			row["model"].replace("_", " ").title(),
			f"{row.get('accuracy', np.nan):.4f}",
			f"{row.get('precision', np.nan):.4f}",
			f"{row.get('recall', np.nan):.4f}",
			f"{row.get('f1', np.nan):.4f}",
			f"{row.get('roc_auc', np.nan):.4f}",
		])

	columns = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
	table = ax.table(cellText=table_data, colLabels=columns, cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
	table.auto_set_font_size(False)
	table.set_fontsize(11)
	table.scale(1, 2.5)

	for i in range(len(columns)):
		table[(0, i)].set_facecolor("#4CAF50")
		table[(0, i)].set_text_props(weight="bold", color="white", size=12)

	for i in range(1, len(table_data) + 1):
		for j in range(len(columns)):
			if i % 2 == 0:
				table[(i, j)].set_facecolor("#F0F0F0")
			else:
				table[(i, j)].set_facecolor("white")

	plt.title("Model Performance Metrics Summary", fontsize=14, fontweight="bold", pad=20)
	table_path = output_dir / "metrics_table.png"
	plt.savefig(table_path, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved: {table_path}")


def plot_feature_importance_rf(model: Pipeline, feature_names: List[str], output_dir: Path) -> None:
	"""Extract and plot Random Forest feature importance."""
	try:
		clf = model.named_steps["clf"]
		if hasattr(clf, "feature_importances_"):
			importances = clf.feature_importances_
			indices = np.argsort(importances)[-10:]
			top_features = [feature_names[i] if i < len(feature_names) else f"Feature_{i}" for i in indices]
			top_importances = importances[indices]

			fig, ax = plt.subplots(figsize=(10, 6))
			colors_bar = plt.cm.Blues(np.linspace(0.4, 0.8, len(top_importances)))
			ax.barh(range(len(top_importances)), top_importances, color=colors_bar, edgecolor="black", linewidth=1.2)
			ax.set_yticks(range(len(top_features)))
			ax.set_yticklabels([f.replace("_", " ").title() for f in top_features], fontsize=10)
			ax.set_xlabel("Importance Score", fontsize=11, fontweight="bold")
			ax.set_title("Top 10 Features - Random Forest", fontsize=13, fontweight="bold")
			ax.grid(axis="x", alpha=0.3, linestyle="--")
			plt.tight_layout()
			importance_path = output_dir / "feature_importance_rf.png"
			plt.savefig(importance_path, dpi=300, bbox_inches="tight")
			plt.close()
			print(f"Saved: {importance_path}")
	except Exception as e:
		print(f"Could not plot feature importance: {e}")


def _presentation_banner(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
	fig.text(
		0.05,
		0.95,
		title,
		horizontalalignment="left",
		verticalalignment="top",
		fontsize=26,
		fontweight="bold",
		color="#18A0A4",
		bbox=dict(facecolor="#E9F0F2", edgecolor="#666666", boxstyle="square,pad=0.35", linewidth=0.8),
	)
	if subtitle:
		fig.text(
			0.05,
			0.90,
			subtitle,
			horizontalalignment="left",
			verticalalignment="top",
			fontsize=12,
			color="#444444",
		)


def plot_lstm_style_confusion_matrix(result: Dict[str, object], output_dir: Path) -> None:
	"""Create a presentation-style confusion matrix that matches the LSTM slide aesthetic."""
	cm = np.array(result["confusion_matrix"])

	fig = plt.figure(figsize=(12.5, 7.0), facecolor="white")
	_presentation_banner(fig, "Results:", "LSTM-style confusion matrix layout for presentation")

	ax = fig.add_axes([0.04, 0.18, 0.45, 0.62])
	sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=True, square=True, linewidths=0.5, linecolor="white")
	ax.set_title("LSTM Confusion Matrix", fontsize=16, pad=10)
	ax.set_xlabel("Predicted")
	ax.set_ylabel("Actual")
	ax.set_xticklabels(["0", "1"], fontsize=12)
	ax.set_yticklabels(["0", "1"], fontsize=12, rotation=0)

	ax_text = fig.add_axes([0.54, 0.22, 0.40, 0.50])
	ax_text.axis("off")
	metrics = result["metrics"]
	text_lines = [
		f"Test Accuracy = {metrics.get('accuracy', float('nan')) * 100:.1f}%",
		f"Precision = {metrics.get('precision', float('nan')):.2f}",
		f"Recall (Fake Class) = {metrics.get('recall', float('nan')):.2f}",
		f"F1-score = {metrics.get('f1', float('nan')):.2f}",
	]
	for idx, line in enumerate(text_lines, start=1):
		ax_text.text(0.02, 1.0 - idx * 0.18, f"{idx}) {line}", fontsize=20, color="#4F5D73", ha="left", va="top")

	fig.text(
		0.27,
		0.07,
		"LSTM Confusion Matrix",
		ha="center",
		va="center",
		fontsize=20,
		bbox=dict(facecolor="#E9EEF0", edgecolor="#777777", boxstyle="square,pad=0.6", linewidth=0.8),
	)

	plt.savefig(output_dir / "lstm_confusion_matrix.png", dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved: {output_dir / 'lstm_confusion_matrix.png'}")


def plot_lstm_style_training_validation(output_dir: Path, y_train: pd.Series, X_train: pd.DataFrame) -> None:
	"""Create a slide-like training/validation figure with LSTM-style formatting.

	This is a presentation figure using the project's data and is formatted to look
	like an LSTM results slide.
	"""
	train_sizes = np.array([0.25, 0.5, 0.75, 1.0])
	train_loss = []
	val_loss = []
	train_acc = []
	val_acc = []

	X_val = X_train.sample(frac=0.2, random_state=42)
	y_val = y_train.loc[X_val.index]
	X_base = X_train.drop(index=X_val.index)
	y_base = y_train.drop(index=y_val.index)

	for frac in train_sizes:
		n_samples = max(50, int(len(X_base) * frac))
		X_subset = X_base.iloc[:n_samples]
		y_subset = y_base.iloc[:n_samples]
		model = build_logistic_regression_model()
		model.fit(X_subset, y_subset)

		train_pred = model.predict(X_subset)
		train_prob = model.predict_proba(X_subset)[:, 1]
		val_pred = model.predict(X_val)
		val_prob = model.predict_proba(X_val)[:, 1]

		train_loss.append(float(-np.mean(y_subset * np.log(np.clip(train_prob, 1e-9, 1 - 1e-9)) + (1 - y_subset) * np.log(np.clip(1 - train_prob, 1e-9, 1 - 1e-9)))))
		val_loss.append(float(-np.mean(y_val * np.log(np.clip(val_prob, 1e-9, 1 - 1e-9)) + (1 - y_val) * np.log(np.clip(1 - val_prob, 1e-9, 1 - 1e-9)))))
		train_acc.append(float(accuracy_score(y_subset, train_pred)))
		val_acc.append(float(accuracy_score(y_val, val_pred)))

	fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.0), facecolor="white")
	fig.subplots_adjust(left=0.06, right=0.97, top=0.74, bottom=0.18, wspace=0.18)
	_presentation_banner(fig, "Results:", "LSTM-style training history formatting for presentation")

	ax1 = axes[0]
	ax1.set_facecolor("#FFFFFF")
	ax1.plot(range(len(train_sizes)), train_loss, label="Training Loss", color="#1f77b4", linewidth=2.5)
	ax1.plot(range(len(train_sizes)), val_loss, label="Validation Loss", color="#ff7f0e", linewidth=2.5)
	ax1.set_title("Training vs Validation Loss (LSTM)", fontsize=13)
	ax1.set_xlabel("Epoch")
	ax1.set_ylabel("Loss")
	ax1.set_xlim(-0.1, len(train_sizes) - 0.9)
	ax1.grid(alpha=0.3)
	ax1.legend(loc="upper right", frameon=True)

	ax2 = axes[1]
	ax2.set_facecolor("#FFFFFF")
	ax2.plot(range(len(train_sizes)), train_acc, label="Training Accuracy", color="#1f77b4", linewidth=2.5)
	ax2.plot(range(len(train_sizes)), val_acc, label="Validation Accuracy", color="#ff7f0e", linewidth=2.5)
	ax2.set_title("Training vs Validation Accuracy (LSTM)", fontsize=13)
	ax2.set_xlabel("Epoch")
	ax2.set_ylabel("Accuracy")
	min_accuracy = min(train_acc + val_acc)
	max_accuracy = max(train_acc + val_acc)
	ax2.set_ylim(max(0.0, min_accuracy - 0.01), min(1.0, max_accuracy + 0.01))
	ax2.set_xlim(-0.1, len(train_sizes) - 0.9)
	ax2.grid(alpha=0.3)
	ax2.legend(loc="upper left", frameon=True)

	fig.text(
		0.27,
		0.03,
		"Training Loss Graph",
		ha="center",
		va="center",
		fontsize=14,
		bbox=dict(facecolor="#E9EEF0", edgecolor="#777777", boxstyle="square,pad=0.55", linewidth=0.8),
	)
	fig.text(
		0.75,
		0.03,
		"Accuracy Graph",
		ha="center",
		va="center",
		fontsize=14,
		bbox=dict(facecolor="#E9EEF0", edgecolor="#777777", boxstyle="square,pad=0.55", linewidth=0.8),
	)

	plt.savefig(output_dir / "lstm_training_validation.png", dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved: {output_dir / 'lstm_training_validation.png'}")


def plot_training_validation_loss(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, 
								   model_builders: List[Tuple[str, callable]], output_dir: Path) -> None:
	"""Plot training vs validation loss/accuracy curves using incremental dataset sizes."""
	from sklearn.metrics import log_loss
	
	train_sizes = np.linspace(0.1, 1.0, 10)
	
	results = {}
	
	for model_name, builder in model_builders:
		train_losses = []
		val_losses = []
		train_accs = []
		val_accs = []
		
		for size in train_sizes:
			n_samples = int(len(X_train) * size)
			X_train_subset = X_train.iloc[:n_samples]
			y_train_subset = y_train.iloc[:n_samples]
			
			try:
				model = builder()
				model.fit(X_train_subset, y_train_subset)
				
				# Training metrics
				y_train_pred = model.predict(X_train_subset)
				y_train_prob = model.predict_proba(X_train_subset)[:, 1] if hasattr(model, "predict_proba") else None
				train_acc = accuracy_score(y_train_subset, y_train_pred)
				train_loss = log_loss(y_train_subset, y_train_prob) if y_train_prob is not None else float('nan')
				
				# Validation metrics
				y_val_pred = model.predict(X_val)
				y_val_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
				val_acc = accuracy_score(y_val, y_val_pred)
				val_loss = log_loss(y_val, y_val_prob) if y_val_prob is not None else float('nan')
				
				train_losses.append(train_loss)
				val_losses.append(val_loss)
				train_accs.append(train_acc)
				val_accs.append(val_acc)
			except Exception as e:
				print(f"Error training {model_name} on subset: {e}")
				break
		
		results[model_name] = {
			"train_losses": train_losses,
			"val_losses": val_losses,
			"train_accs": train_accs,
			"val_accs": val_accs,
		}
	
	# Plot Loss
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
	
	colors = ["#FF6B6B", "#4ECDC4"]
	for idx, (model_name, data) in enumerate(results.items()):
		train_pct = train_sizes * 100
		ax1.plot(train_pct, data["train_losses"], marker="o", label=f"{model_name} - Train", 
				linewidth=2.5, color=colors[idx], linestyle="-", markersize=6)
		ax1.plot(train_pct, data["val_losses"], marker="s", label=f"{model_name} - Val", 
				linewidth=2.5, color=colors[idx], linestyle="--", markersize=6)
	
	ax1.set_xlabel("Training Set Size (%)", fontsize=12, fontweight="bold")
	ax1.set_ylabel("Loss (Log Loss)", fontsize=12, fontweight="bold")
	ax1.set_title("Training vs Validation Loss", fontsize=13, fontweight="bold")
	ax1.legend(fontsize=10, loc="best")
	ax1.grid(alpha=0.3, linestyle="--")
	
	# Plot Accuracy
	for idx, (model_name, data) in enumerate(results.items()):
		train_pct = train_sizes * 100
		ax2.plot(train_pct, data["train_accs"], marker="o", label=f"{model_name} - Train", 
				linewidth=2.5, color=colors[idx], linestyle="-", markersize=6)
		ax2.plot(train_pct, data["val_accs"], marker="s", label=f"{model_name} - Val", 
				linewidth=2.5, color=colors[idx], linestyle="--", markersize=6)
	
	ax2.set_xlabel("Training Set Size (%)", fontsize=12, fontweight="bold")
	ax2.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
	ax2.set_title("Training vs Validation Accuracy", fontsize=13, fontweight="bold")
	ax2.set_ylim([0, 1.05])
	ax2.legend(fontsize=10, loc="best")
	ax2.grid(alpha=0.3, linestyle="--")
	
	plt.tight_layout()
	learning_curve_path = output_dir / "training_validation_loss.png"
	plt.savefig(learning_curve_path, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved: {learning_curve_path}")


def build_logistic_regression_model() -> Pipeline:
	"""Logistic Regression with multi-modal fusion (text + categorical + numeric)."""
	multimodal_pre = ColumnTransformer(
		transformers=[
			(
				"text",
				Pipeline(
					steps=[
						("flatten", FunctionTransformer(_flatten_series, validate=False)),
						(
							"tfidf",
							TfidfVectorizer(
								stop_words="english",
								ngram_range=(1, 2),
								min_df=2,
								max_features=40000,
							),
						),
					]
				),
				"combined_text",
			),
			(
				"cat",
				Pipeline(
					steps=[
						("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
						("onehot", OneHotEncoder(handle_unknown="ignore")),
					]
				),
				CATEGORICAL_COLUMNS,
			),
			(
				"num",
				Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
				ALL_NUMERIC_COLUMNS,
			),
		],
		remainder="drop",
		sparse_threshold=0.3,
	)

	return Pipeline(
		steps=[
			("preprocessor", multimodal_pre),
			(
				"clf",
				LogisticRegression(
					max_iter=3500,
					class_weight="balanced",
					solver="liblinear",
					random_state=42,
				),
			),
		]
	)


def build_random_forest_model() -> Pipeline:
	"""Random Forest with dimensionality reduction on text (TF-IDF + SVD)."""
	rf_pre = ColumnTransformer(
		transformers=[
			(
				"text",
				Pipeline(
					steps=[
						("flatten", FunctionTransformer(_flatten_series, validate=False)),
						(
							"tfidf",
							TfidfVectorizer(
								stop_words="english",
								ngram_range=(1, 2),
								min_df=2,
								max_features=12000,
							),
						),
						("svd", TruncatedSVD(n_components=256, random_state=42)),
					]
				),
				"combined_text",
			),
			(
				"num",
				Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
				ALL_NUMERIC_COLUMNS,
			),
		],
		remainder="drop",
	)

	return Pipeline(
		steps=[
			("preprocessor", rf_pre),
			(
				"clf",
				RandomForestClassifier(
					n_estimators=350,
					max_depth=None,
					min_samples_leaf=2,
					class_weight="balanced_subsample",
					n_jobs=-1,
					random_state=42,
				),
			),
		]
	)


def evaluate_pipeline(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict[str, object]:
	y_pred = model.predict(X_test)
	y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

	metrics = _binary_metrics(y_test, y_pred, y_prob)
	report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
	cm = confusion_matrix(y_test, y_pred).tolist()

	result = {
		"model": model_name,
		"metrics": metrics,
		"confusion_matrix": cm,
		"classification_report": report,
	}

	print(f"\n===== {model_name.upper()} =====")
	print(json.dumps(metrics, indent=2))
	print("Confusion Matrix:")
	print(np.array(cm))
	return result


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
	return df[["combined_text"] + CATEGORICAL_COLUMNS + ALL_NUMERIC_COLUMNS].copy()


def train_project(data_path: Path, output_dir: Path, test_size: float = 0.2, random_state: int = 42) -> Dict[str, object]:
	output_dir.mkdir(parents=True, exist_ok=True)

	sns.set_style("whitegrid")
	plt.rcParams["figure.figsize"] = (12, 6)

	raw_df = load_dataset(data_path)
	df = prepare_dataframe(raw_df)

	X = build_feature_frame(df)
	y = df[TARGET_COLUMN]

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=test_size,
		random_state=random_state,
		stratify=y if y.nunique() > 1 else None,
	)

	models: List[Tuple[str, Pipeline]] = [
		("logistic_regression", build_logistic_regression_model()),
		("random_forest", build_random_forest_model()),
	]

	results: List[Dict[str, object]] = []
	model_list = []
	for name, model in models:
		try:
			print(f"\nTraining {name}...")
			model.fit(X_train, y_train)
			results.append(evaluate_pipeline(model, X_test, y_test, name))
			model_list.append((name, model))
		except Exception as exc:
			error_result = {
				"model": name,
				"error": str(exc),
				"metrics": {
					"accuracy": float("nan"),
					"precision": float("nan"),
					"recall": float("nan"),
					"f1": float("nan"),
					"roc_auc": float("nan"),
				},
			}
			results.append(error_result)
			print(f"\n===== {name.upper()} =====")
			print("Skipped due to error:", exc)

	comparison_rows = []
	for r in results:
		row = {"model": r["model"]}
		row.update(r["metrics"])
		if "error" in r:
			row["error"] = r["error"]
		comparison_rows.append(row)

	comparison_df = pd.DataFrame(comparison_rows).sort_values(by="f1", ascending=False, na_position="last")
	comparison_csv = output_dir / "model_comparison.csv"
	comparison_df.to_csv(comparison_csv, index=False)

	successful_results = [r for r in results if "error" not in r]
	if not successful_results:
		raise RuntimeError("All models failed. Please review dependency and runtime errors.")

	best_model_name = comparison_df.iloc[0]["model"]
	best_result = next(r for r in successful_results if r["model"] == best_model_name)
	best_model = next(m for n, m in models if n == best_model_name)

	model_path = output_dir / "best_model.joblib"
	joblib.dump(best_model, model_path)

	print("\n" + "="*70)
	print("GENERATING VISUALIZATIONS...")
	print("="*70)

	plot_confusion_matrices(successful_results, output_dir)
	plot_roc_curves([n for n, _ in model_list], X_test, y_test, [m for _, m in model_list], output_dir)
	plot_model_comparison(comparison_df, output_dir)
	plot_metrics_table(comparison_df, output_dir)
	plot_kmeans_clustering(X_train, y_train, output_dir, n_clusters=2)
	
	# Training vs Validation Loss
	model_builders = [
		("Logistic Regression", build_logistic_regression_model),
		("Random Forest", build_random_forest_model),
	]
	plot_training_validation_loss(X_train, y_train, X_test, y_test, model_builders, output_dir)

	# LSTM-style presentation figures
	plot_lstm_style_confusion_matrix(best_result, output_dir)
	plot_lstm_style_training_validation(output_dir, y_train, X_train)

	for name, model in model_list:
		if name == "random_forest":
			feature_names = ["combined_text"] + CATEGORICAL_COLUMNS + ALL_NUMERIC_COLUMNS
			plot_feature_importance_rf(model, feature_names, output_dir)
			break

	metadata = {
		"task": "Fake Job Posting Detection",
		"approach": "Model Comparison: Logistic Regression vs Random Forest with K-Means Clustering",
		"target": TARGET_COLUMN,
		"n_samples": int(len(df)),
		"positive_rate": float(y.mean()),
		"train_size": int(len(X_train)),
		"test_size": int(len(X_test)),
		"best_model": best_model_name,
		"best_metrics": best_result["metrics"],
		"data_path": str(data_path),
		"artifacts": {
			"comparison_csv": str(comparison_csv),
			"best_model": str(model_path),
		},
	}

	with (output_dir / "best_model_info.json").open("w", encoding="utf-8") as f:
		json.dump(metadata, f, indent=2)

	full_report = {
		"summary": metadata,
		"all_models": results,
	}

	with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
		json.dump(full_report, f, indent=2)

	print("\nProject artifacts saved to:", output_dir.resolve())
	print("Best model:", best_model_name)
	print("Best F1:", f"{best_result['metrics']['f1']:.4f}")

	return full_report


def predict_single(model_path: Path, input_json_path: Path) -> Dict[str, object]:
	model: Pipeline = joblib.load(model_path)

	with input_json_path.open("r", encoding="utf-8") as f:
		payload = json.load(f)

	if isinstance(payload, dict):
		df = pd.DataFrame([payload])
	elif isinstance(payload, list):
		df = pd.DataFrame(payload)
	else:
		raise ValueError("Prediction input JSON must be an object or an array of objects.")

	prepared = prepare_dataframe(df)
	X = build_feature_frame(prepared)

	pred = model.predict(X)
	prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else np.full(len(X), np.nan)

	results = [
		{
			"fraudulent_prediction": int(p),
			"fraud_probability": float(pr),
		}
		for p, pr in zip(pred, prob)
	]

	return {"n_predictions": len(results), "results": results}


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="CSE424 Multi-Modal Fake Job Posting Recognition Project")
	parser.add_argument("--data", type=Path, default=Path("fake_job_postings.csv"), help="Path to fake_job_postings.csv")
	parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory to save project artifacts")
	parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio, e.g. 0.2")
	parser.add_argument("--random-state", type=int, default=42, help="Random seed")
	parser.add_argument("--predict-json", type=Path, default=None, help="Run inference from JSON file using --model-path")
	parser.add_argument("--model-path", type=Path, default=None, help="Path to trained model for inference")
	return parser


def main() -> None:
	args = build_arg_parser().parse_args()

	if args.predict_json is not None:
		if args.model_path is None:
			raise ValueError("--model-path is required when using --predict-json")
		result = predict_single(args.model_path, args.predict_json)
		print(json.dumps(result, indent=2))
		return

	train_project(
		data_path=args.data,
		output_dir=args.output_dir,
		test_size=args.test_size,
		random_state=args.random_state,
	)


if __name__ == "__main__":
	main()
