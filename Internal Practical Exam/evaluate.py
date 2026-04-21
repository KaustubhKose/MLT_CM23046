"""
evaluate.py — Comprehensive evaluation of the trained fruit classifier
Generates:
  • Classification report (precision / recall / F1)
  • Confusion matrix heatmap
  • Per-class accuracy bar chart
  • Sample prediction grid

Usage:
    python evaluate.py
    python evaluate.py --model models/fruit_classifier_final.keras --data data/val
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
def load_resources(model_path: str, class_map_path: str):
    model = tf.keras.models.load_model(model_path)
    with open(class_map_path) as f:
        class_map = {int(k): v for k, v in json.load(f).items()}
    return model, class_map


def build_val_generator(val_dir: str, img_size=(224, 224), batch_size=32,
                        classes=None):
    gen = ImageDataGenerator(rescale=1.0 / 255)
    return gen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=classes,
        shuffle=False,
    )


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
def run_evaluation(model, val_gen, class_map):
    class_names = [class_map[i] for i in sorted(class_map)]

    print("⏳ Running predictions on validation set…")
    y_pred_probs = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes

    print("\n─── Classification Report ──────────────────────────")
    print(classification_report(y_true, y_pred, target_names=class_names))

    return y_true, y_pred, y_pred_probs, class_names


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2%"],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalised)"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True",      fontsize=11)
        ax.set_title(title,        fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Confusion matrix saved → {path}")


def plot_per_class_accuracy(y_true, y_pred, class_names, save_dir):
    accs = []
    for i, name in enumerate(class_names):
        mask = y_true == i
        accs.append(np.mean(y_pred[mask] == i) if mask.sum() > 0 else 0.0)

    colors = ["#4CAF50" if a >= 0.9 else "#FF9800" if a >= 0.7 else "#F44336"
              for a in accs]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(class_names, [a * 100 for a in accs], color=colors, width=0.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Per-Class Accuracy", fontsize=13, fontweight="bold")
    ax.axhline(90, color="green",  linestyle="--", alpha=0.5, label="90% line")
    ax.axhline(70, color="orange", linestyle="--", alpha=0.5, label="70% line")
    ax.legend()

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{acc * 100:.1f}%", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(save_dir, "per_class_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Per-class accuracy saved → {path}")


def plot_sample_predictions(model, val_gen, class_map, save_dir, n=12):
    """Show a grid of sample images with predicted vs true labels."""
    class_names = [class_map[i] for i in sorted(class_map)]
    images, labels = [], []
    for imgs, lbls in val_gen:
        images.append(imgs)
        labels.append(lbls)
        if sum(len(b) for b in images) >= n:
            break

    images = np.concatenate(images)[:n]
    labels = np.concatenate(labels)[:n]
    preds  = model.predict(images, verbose=0)

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten()

    for i, (img, lbl, pred) in enumerate(zip(images, labels, preds)):
        true_cls = class_names[np.argmax(lbl)]
        pred_cls = class_names[np.argmax(pred)]
        conf     = pred.max() * 100

        axes[i].imshow(img)
        axes[i].axis("off")
        color = "green" if true_cls == pred_cls else "red"
        axes[i].set_title(
            f"True: {true_cls}\nPred: {pred_cls} ({conf:.0f}%)",
            fontsize=9, color=color, fontweight="bold",
        )

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Sample Predictions  (green=correct, red=wrong)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, "sample_predictions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Sample predictions saved → {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="models/fruit_classifier_final.keras")
    parser.add_argument("--class_map", default="models/class_map.json")
    parser.add_argument("--data",      default="data/val")
    parser.add_argument("--out",       default="models")
    args = parser.parse_args()

    model, class_map = load_resources(args.model, args.class_map)
    classes = [class_map[i] for i in sorted(class_map)]

    val_gen = build_val_generator(args.data, classes=classes)
    y_true, y_pred, y_probs, class_names = run_evaluation(model, val_gen, class_map)

    os.makedirs(args.out, exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, class_names, args.out)
    plot_per_class_accuracy(y_true, y_pred, class_names, args.out)
    plot_sample_predictions(model, val_gen, class_map, args.out)

    print(f"\n✅ Evaluation complete. All charts saved to → {args.out}/")


if __name__ == "__main__":
    main()
