"""
Transfer Learning with MobileNetV2 for Fruit Classification
============================================================
Retrains MobileNetV2 (pretrained on ImageNet) on custom fruit categories.
Supports: apple, banana, orange (easily extendable to more classes).
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "data_dir":       "data",
    "model_dir":      "models",
    "img_size":       (224, 224),
    "batch_size":     32,
    "epochs_frozen":  10,   # Phase 1: only train head
    "epochs_finetune":15,   # Phase 2: fine-tune top layers
    "learning_rate":  1e-3,
    "finetune_lr":    1e-5,
    "unfreeze_layers":30,   # How many top MobileNet layers to unfreeze
    "classes":        ["apple", "banana", "orange"],
}

# ─────────────────────────────────────────────
# DATA PIPELINE
# ─────────────────────────────────────────────
def build_data_generators(cfg):
    """Create augmented train and validation data generators."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.10,
        zoom_range=0.20,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(cfg["data_dir"], "train"),
        target_size=cfg["img_size"],
        batch_size=cfg["batch_size"],
        class_mode="categorical",
        classes=cfg["classes"],
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        os.path.join(cfg["data_dir"], "val"),
        target_size=cfg["img_size"],
        batch_size=cfg["batch_size"],
        class_mode="categorical",
        classes=cfg["classes"],
        shuffle=False,
    )

    return train_gen, val_gen


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def build_model(num_classes: int, img_size: tuple) -> Model:
    """
    Transfer Learning architecture:
      MobileNetV2 (frozen) → GlobalAvgPool → Dense(256) → Dropout → Softmax
    """
    base_model = MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Freeze all base layers initially

    inputs = tf.keras.Input(shape=(*img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model, base_model


def unfreeze_top_layers(model, base_model, n_layers: int, lr: float):
    """Phase 2: Unfreeze top N layers of base model for fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"\n✅ Unfroze top {n_layers} layers for fine-tuning (lr={lr})")
    return model


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def get_callbacks(model_dir: str, phase: str):
    os.makedirs(model_dir, exist_ok=True)
    return [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, f"best_{phase}.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
        callbacks.TensorBoard(
            log_dir=os.path.join(model_dir, "logs", phase),
        ),
    ]


def plot_history(history_frozen, history_finetune, save_path="models/training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Combine histories
    acc  = history_frozen.history["accuracy"]      + history_finetune.history["accuracy"]
    val_acc = history_frozen.history["val_accuracy"] + history_finetune.history["val_accuracy"]
    loss = history_frozen.history["loss"]          + history_finetune.history["loss"]
    val_loss = history_frozen.history["val_loss"]  + history_finetune.history["val_loss"]
    split = len(history_frozen.history["accuracy"])

    for ax, (train, val, title) in zip(axes, [
        (acc, val_acc, "Accuracy"),
        (loss, val_loss, "Loss"),
    ]):
        ax.plot(train, label="Train", linewidth=2)
        ax.plot(val,   label="Val",   linewidth=2)
        ax.axvline(x=split, color="gray", linestyle="--", label="Fine-tune start")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"📊 Training curves saved → {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    cfg = CONFIG
    num_classes = len(cfg["classes"])

    print("🍎 Transfer Learning — Fruit Classifier")
    print(f"   Classes   : {cfg['classes']}")
    print(f"   Image size: {cfg['img_size']}")
    print(f"   Batch size: {cfg['batch_size']}\n")

    train_gen, val_gen = build_data_generators(cfg)
    model, base_model = build_model(num_classes, cfg["img_size"])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ── Phase 1: Train head only ──────────────────
    print("\n📌 Phase 1: Training classification head (base frozen)…")
    hist1 = model.fit(
        train_gen,
        epochs=cfg["epochs_frozen"],
        validation_data=val_gen,
        callbacks=get_callbacks(cfg["model_dir"], "phase1"),
    )

    # ── Phase 2: Fine-tune top layers ────────────
    print("\n📌 Phase 2: Fine-tuning top MobileNet layers…")
    model = unfreeze_top_layers(model, base_model, cfg["unfreeze_layers"], cfg["finetune_lr"])
    hist2 = model.fit(
        train_gen,
        epochs=cfg["epochs_finetune"],
        validation_data=val_gen,
        callbacks=get_callbacks(cfg["model_dir"], "phase2"),
    )

    # ── Save final model & class map ─────────────
    final_path = os.path.join(cfg["model_dir"], "fruit_classifier_final.keras")
    model.save(final_path)
    print(f"\n✅ Final model saved → {final_path}")

    class_map = {v: k for k, v in train_gen.class_indices.items()}
    with open(os.path.join(cfg["model_dir"], "class_map.json"), "w") as f:
        json.dump(class_map, f, indent=2)
    print("📝 Class map saved → models/class_map.json")

    plot_history(hist1, hist2)

    # ── Final evaluation ─────────────────────────
    print("\n📊 Final validation metrics:")
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"   Loss    : {val_loss:.4f}")
    print(f"   Accuracy: {val_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
