"""This code re-implements the tutorial on https://www.tensorflow.org/tutorials/interpretability/integrated_gradients."""

from __future__ import absolute_import, print_function

import argparse
import os
from typing import List, Union

import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from matplotlib.figure import Figure


def plot_img_attributions(
    baseline: tf.Tensor,
    image: tf.Tensor,
    attributions: tf.Tensor,
    cmap: plt.cm = None,
    overlay_alpha: float = 0.4,
) -> Figure:
    """Plot Attributions.

    Args:
        baseline (tf.Tensor): baseline image.
        image (tf.Tensor): target image.
        attributions (tf.Tensor): attributions.
        cmap (plt.cm, optional): colormap. Defaults to None.
        overlay_alpha (float, optional): transparency. Defaults to 0.4.

    Returns:
        Figure: attribution figure.
    """
    # Sum of the attributions across color channels for visualization.
    # The attribution mask shape is a grayscale image with height and width
    # equal to the original image.
    attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

    fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))

    axs[0, 0].set_title("Baseline image")
    axs[0, 0].imshow(baseline)
    axs[0, 0].axis("off")

    axs[0, 1].set_title("Original image")
    axs[0, 1].imshow(image)
    axs[0, 1].axis("off")

    axs[1, 0].set_title("Attribution mask")
    axs[1, 0].imshow(attribution_mask, cmap=cmap)
    axs[1, 0].axis("off")

    axs[1, 1].set_title("Overlay")
    axs[1, 1].imshow(attribution_mask, cmap=cmap)
    axs[1, 1].imshow(image, alpha=overlay_alpha)
    axs[1, 1].axis("off")

    plt.tight_layout()
    return fig


@tf.function
def one_batch(
    baseline: tf.Tensor,
    image: tf.Tensor,
    alpha_batch: tf.Tensor,
    target_class_idx: int,
    model: tf.keras.Model,
) -> tf.Tensor:
    """Run One Batch.

    Args:
        baseline (tf.Tensor): baseline image.
        image (tf.Tensor): target image
        alpha_batch (tf.Tensor): interpolation step sizes .
        target_class_idx (int): target class.
        model (tf.keras.Model): classifier.

    Returns:
        tf.Tensor: gradients.
    """
    # Generate interpolated inputs between baseline and input.
    interpolated_path_input_batch = interpolate_images(
        baseline=baseline, image=image, alphas=alpha_batch
    )

    # Compute gradients between model outputs and interpolated inputs.
    gradient_batch = compute_gradients(
        images=interpolated_path_input_batch,
        target_class_idx=target_class_idx,
        model=model,
    )
    return gradient_batch


def integrated_gradients(
    baseline: tf.Tensor,
    image: tf.Tensor,
    target_class_idx: int,
    alphas: tf.Tensor,
    model: tf.keras.Model,
    m_steps: int = 50,
    batch_size: int = 32,
) -> tf.Tensor:
    """Compute Integrated Gradients.

    Args:
        baseline (tf.Tensor): tf.Tensor.
        image (tf.Tensor): tf.Tensor.
        target_class_idx (int): target class.
        m_steps (int, optional): number of steps. Defaults to 50.
        batch_size (int, optional): batch size. Defaults to 32.

    Returns:
        tf.Tensor: integrated gradients.
    """
    # Generate alphas.
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)

    # Collect gradients.
    gradient_batches = []

    # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]

        gradient_batch = one_batch(
            baseline, image, alpha_batch, target_class_idx, model
        )
        gradient_batches.append(gradient_batch)

    # Concatenate path gradients together row-wise into single tensor.
    total_gradients = tf.concat(gradient_batches, axis=0)

    # Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(gradients=total_gradients)

    # Scale integrated gradients with respect to input.
    integrated_gradients = (image - baseline) * avg_gradients

    return integrated_gradients


def integral_approximation(gradients: tf.Tensor) -> tf.Tensor:
    """Approximate Integrals with Riemann Sum.

    Args:
        gradients (tf.Tensor): gradients.

    Returns:
        tf.Tensor: integral of gradients.
    """
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients


def compute_gradients(
    images: tf.Tensor, target_class_idx: int, model: tf.keras.Model
) -> tf.Tensor:
    """Compute Gradients using Classifier and Label.

    Args:
        images (tf.Tensor): input image.
        target_class_idx (int): true label.
        model (tf.keras.Model): classifier.

    Returns:
        tf.Tensor: gradients.
    """
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = model(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, images)


def interpolate_images(
    baseline: tf.Tensor, image: tf.Tensor, alphas: tf.Tensor
) -> List[tf.Tensor]:
    """Interpolate Images Given Step Size.

    Args:
        baseline (tf.Tensor): baseline image.
        image (tf.Tensor): target image.
        alphas (tf.Tensor): step size.

    Returns:
        List[tf.Tensor]: intepolated images.
    """
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x + alphas_x * delta
    return images


def top_k_predictions(
    img: tf.Tensor, model: tf.keras.Model, imagenet_labels: np.array, k: int = 3
) -> Union[int, float]:
    """Classify Input Image and Return Top K Predictions.

    Args:
        img (tf.Tensor): input image.
        model (tf.keras.Model): classifier.
        imagenet_labels (np.array): input labels.
        k (int, optional): k. Defaults to 3.

    Returns:
        Union[int, float]: label and probability.
    """
    image_batch = tf.expand_dims(img, 0)
    predictions = model(image_batch)
    probs = tf.nn.softmax(predictions, axis=-1)
    top_probs, top_idxs = tf.math.top_k(input=probs, k=k)
    top_labels = imagenet_labels[tuple(top_idxs)]
    return top_labels, top_probs[0]


def read_image(file_name: str) -> tf.Tensor:
    """Read Image from Given Path.

    Args:
        file_name (str): image path.

    Returns:
        tf.Tensor: image.
    """
    image = tf.io.read_file(file_name)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_pad(image, target_height=224, target_width=224)
    return image


def load_imagenet_labels(file_path: str) -> np.array:
    """Load Labels from Given Path.

    Args:
        file_path (str): path to labels.

    Returns:
        np.array: labels
    """
    labels_file = tf.keras.utils.get_file("ImageNetLabels.txt", file_path)
    with open(labels_file) as reader:
        f = reader.read()
        labels = f.splitlines()
    return np.array(labels)


def get_model() -> tf.keras.Model:
    """Get pretrained model.

    Returns:
        tf.keras.Model: Inception V1 model.
    """
    model = tf.keras.Sequential(
        [
            hub.KerasLayer(
                name="inception_v1",
                handle="https://tfhub.dev/google/imagenet/inception_v1/classification/4",
                trainable=False,
            )
        ]
    )
    model.build([None, 224, 224, 3])
    model.summary()
    return model


def main(args: argparse.Namespace) -> None:
    """Run Integrated Gradients.

    Args:
        args (argparse.Namespace): input arguments.
    """
    os.makedirs(args.image_dir, exist_ok=True)
    model = get_model()
    imagenet_labels = load_imagenet_labels(
        "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
    )

    # Load two images and classify
    img_url = {
        "Fireboat": "http://storage.googleapis.com/download.tensorflow.org/example_images/San_Francisco_fireboat_showing_off.jpg",
        "Giant Panda": "http://storage.googleapis.com/download.tensorflow.org/example_images/Giant_Panda_2.jpeg",
    }

    img_paths = {
        name: tf.keras.utils.get_file(name, url) for (name, url) in img_url.items()
    }
    img_name_tensors = {
        name: read_image(img_path) for (name, img_path) in img_paths.items()
    }

    f = plt.figure(figsize=(8, 8))
    for n, (name, img_tensors) in enumerate(img_name_tensors.items()):
        plt.subplot(1, 2, n + 1)
        plt.imshow(img_tensors)
        plt.title(name)
        plt.axis("off")
    plt.tight_layout()
    f.savefig(os.path.join(args.image_dir, "original.jpg"))

    f = plt.figure(figsize=(8, 8))
    for n, (name, img_tensor) in enumerate(img_name_tensors.items()):
        pred_label, pred_prob = top_k_predictions(img_tensor, model, imagenet_labels)
        pred_str = ""
        for label, prob in zip(pred_label, pred_prob):
            pred_str += f"{label}: {prob:0.1%}\n"

        plt.subplot(1, 2, n + 1)
        plt.imshow(img_tensor)
        plt.title(f"{name} \n {pred_str}", fontweight="bold")
        plt.axis("off")
    plt.tight_layout()
    f.savefig(os.path.join(args.image_dir, "prediction.jpg"))

    # Establish baseline and interpolate
    baseline = tf.zeros(shape=(224, 224, 3))
    f = plt.figure(figsize=(8, 8))
    plt.subplot(1, 1, 1)
    plt.imshow(baseline)
    plt.title("Baseline")
    plt.axis("off")
    plt.tight_layout()
    f.savefig(os.path.join(args.image_dir, "baseline.jpg"))

    # Generate m_steps intervals for integral_approximation() below.
    m_steps = 50

    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)
    interpolated_images = interpolate_images(
        baseline=baseline, image=img_name_tensors["Fireboat"], alphas=alphas
    )

    f = plt.figure(figsize=(20, 20))
    i = 0
    for alpha, image in zip(alphas[0::10], interpolated_images[0::10]):
        i += 1
        plt.subplot(2, len(alphas[0::10]), i)
        plt.title(f"alpha: {alpha:.1f}")
        plt.imshow(image)
        plt.axis("off")

    interpolated_images = interpolate_images(
        baseline=baseline, image=img_name_tensors["Giant Panda"], alphas=alphas
    )

    for alpha, image in zip(alphas[0::10], interpolated_images[0::10]):
        i += 1
        plt.subplot(2, len(alphas[0::10]), i)
        plt.title(f"alpha: {alpha:.1f}")
        plt.imshow(image)
        plt.axis("off")

    plt.tight_layout()
    f.savefig(os.path.join(args.image_dir, "interpolate.jpg"))

    # Compute Gradients and Plot Attributions.
    attributions = integrated_gradients(
        baseline=baseline,
        image=img_name_tensors["Fireboat"],
        target_class_idx=555,
        alphas=alphas,
        model=model,
        m_steps=240,
    )

    f = plot_img_attributions(
        baseline=baseline,
        image=img_name_tensors["Fireboat"],
        attributions=attributions,
        cmap=plt.cm.inferno,
        overlay_alpha=0.4,
    )

    f.savefig(os.path.join(args.image_dir, "fireboat.jpg"))

    attributions = integrated_gradients(
        baseline=baseline,
        image=img_name_tensors["Giant Panda"],
        target_class_idx=389,
        alphas=alphas,
        model=model,
        m_steps=55,
    )

    f = plot_img_attributions(
        baseline=baseline,
        image=img_name_tensors["Giant Panda"],
        attributions=attributions,
        cmap=plt.cm.viridis,
        overlay_alpha=0.5,
    )

    f.savefig(os.path.join(args.image_dir, "panda.jpg"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="./images")
    main(parser.parse_args())
