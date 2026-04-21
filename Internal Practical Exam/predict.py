"""
predict.py — Run inference with the trained fruit classifier
Usage:
    python predict.py --image path/to/fruit.jpg
    python predict.py --image path/to/fruit.jpg --model models/fruit_classifier_final.keras
"""

import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────
# LOADER
# ─────────────────────────────────────────────
def load_model_and_classes(model_path: str, class_map_path: str):
    model = tf.keras.models.load_model(model_path)
    with open(class_map_path) as f:
        class_map = json.load(f)          # {0: "apple", 1: "banana", ...}
    class_map = {int(k): v for k, v in class_map.items()}
    return model, class_map


# ─────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────
def preprocess(img_path: str, img_size=(224, 224)) -> np.ndarray:
    img = keras_image.load_img(img_path, target_size=img_size)
    arr = keras_image.img_to_array(img) / 255.0   # Normalize to [0,1]
    return np.expand_dims(arr, axis=0)             # Add batch dim


# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
def predict(model, img_path: str, class_map: dict, top_k: int = 3):
    x = preprocess(img_path)
    probs = model.predict(x, verbose=0)[0]         # Shape: (num_classes,)

    top_indices = np.argsort(probs)[::-1][:top_k]
    results = [(class_map[i], float(probs[i])) for i in top_indices]

    return results


# ─────────────────────────────────────────────
# VISUALISE
# ─────────────────────────────────────────────
COLORS = ["#4CAF50", "#FF9800", "#2196F3", "#9C27B0", "#F44336"]

def visualise(img_path: str, results: list):
    """Show the image alongside a horizontal bar chart of top-k probabilities."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={"width_ratios": [1, 1.2]})

    # Left: original image
    img = keras_image.load_img(img_path)
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title("Input Image", fontsize=13, fontweight="bold")

    # Right: probability bars
    labels = [r[0].capitalize() for r in results]
    scores = [r[1] * 100 for r in results]
    colors = COLORS[:len(results)]

    bars = axes[1].barh(labels[::-1], scores[::-1], color=colors[::-1], height=0.5)
    axes[1].set_xlim(0, 105)
    axes[1].set_xlabel("Confidence (%)", fontsize=11)
    axes[1].set_title("Predictions", fontsize=13, fontweight="bold")

    for bar, score in zip(bars, scores[::-1]):
        axes[1].text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}%",
            va="center", fontsize=11, fontweight="bold",
        )

    top_label, top_conf = results[0]
    fig.suptitle(
        f"Prediction: {top_label.upper()}  ({top_conf * 100:.1f}% confidence)",
        fontsize=15, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    out_path = "prediction_result.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"📊 Result image saved → {out_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fruit classifier inference")
    parser.add_argument("--image",     required=True,  help="Path to input image")
    parser.add_argument("--model",     default="models/fruit_classifier_final.keras")
    parser.add_argument("--class_map", default="models/class_map.json")
    parser.add_argument("--top_k",     type=int, default=3)
    args = parser.parse_args()

    print(f"🔍 Loading model: {args.model}")
    model, class_map = load_model_and_classes(args.model, args.class_map)

    print(f"🍊 Running inference on: {args.image}")
    results = predict(model, args.image, class_map, top_k=args.top_k)

    print("\n─── Top Predictions ───────────────────")
    for rank, (label, prob) in enumerate(results, 1):
        bar = "█" * int(prob * 30)
        print(f"  {rank}. {label:<12} {bar} {prob * 100:.2f}%")
    print("───────────────────────────────────────\n")

    visualise(args.image, results)


if __name__ == "__main__":
    main()
