"""This code follows the tutorial on https://keras.io/examples/vision/integrated_gradients/."""

from __future__ import absolute_import, print_function

import argparse
import os
from shutil import copyfile
from typing import List, Tuple
from xmlrpc.client import Boolean

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import ndimage
from tensorflow import keras
from tensorflow.keras.applications import xception


def main(args: argparse.Namespace):
    """Run Integrated Gradients.

    Args:
        args (argparse.Namespace): input arguments.
    """
    os.makedirs(args.image_dir, exist_ok=True)

    img_size = (299, 299, 3)

    model = xception.Xception(weights="imagenet")

    img_path = keras.utils.get_file("elephant.jpg", "https://i.imgur.com/Bvro0YD.png")
    copyfile(img_path, os.path.join(args.image_dir, "elephant.jpg"))

    img = get_img_array(img_path)
    orig_img = np.copy(img[0]).astype(np.uint8)
    img_processed = tf.cast(xception.preprocess_input(img), dtype=tf.float32)
    preds = model.predict(img_processed)
    top_pred_idx = tf.argmax(preds[0])
    print(f"Predicted: {top_pred_idx, xception.decode_predictions(preds, top=1)[0]}")

    grads = get_gradients(img_processed, top_pred_idx, model)
    igrads = random_baseline_integrated_gradients(
        model,
        img_size,
        np.copy(orig_img),
        top_pred_idx=top_pred_idx,
        num_steps=50,
        num_runs=2,
    )

    vis = GradVisualizer()

    vis.visualize(
        image=orig_img,
        gradients=grads[0].numpy(),
        integrated_gradients=igrads.numpy(),
        clip_above_percentile=99,
        clip_below_percentile=0,
        output_file=os.path.join(args.image_dir, "integrated_gradients_0_99.jpg"),
    )

    vis.visualize(
        image=orig_img,
        gradients=grads[0].numpy(),
        integrated_gradients=igrads.numpy(),
        clip_above_percentile=95,
        clip_below_percentile=28,
        morphological_cleanup=True,
        outlines=True,
        output_file=os.path.join(args.image_dir, "integrated_gradients_28_95.jpg"),
    )

    vis.visualize(
        image=orig_img,
        gradients=grads[0].numpy(),
        integrated_gradients=igrads.numpy(),
        clip_above_percentile=95,
        clip_below_percentile=28,
        morphological_cleanup=True,
        output_file=os.path.join(args.image_dir, "integrated_gradients_28_95_m.jpg"),
    )


def get_img_array(img_path: str, size: Tuple = (299, 299)) -> np.array:
    """Get image array given image path.

    Args:
        img_path (str): image path.
        size (Tuple, optional): image size. Defaults to (299, 299).

    Returns:
        np.array: image array.
    """
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


def get_gradients(
    img_input: tf.Tensor, top_pred_idx: int, model: tf.keras.Model
) -> List:
    """Compute gradients.

    Args:
        img_input (tf.Tensor): input image.
        top_pred_idx (int): index pf prediction.
        model (tf.keras.Model): classification model

    Returns:
        List: a list or nested structure of Tensors (or IndexedSlices, or None), one for each element in sources. Returned structure is the same as the structure of sources.
    """
    images = tf.cast(img_input, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(images)
        preds = model(images)
        top_class = preds[:, top_pred_idx]

    grads = tape.gradient(top_class, images)
    return grads


def get_integrated_gradients(
    model: tf.keras.Model,
    img_size: Tuple,
    img_input: tf.Tensor,
    top_pred_idx: int,
    baseline: np.array = None,
    num_steps: int = 50,
) -> tf.Tensor:
    """Compute integrated gradients.

    Args:
        model (tf.keras.Model): classfication model.
        img_size (Tuple): image size.
        img_input (tf.Tensor): input image.
        top_pred_idx (int): prediction index.
        baseline (np.array, optional): baseline image. Defaults to None.
        num_steps (int, optional): number of steps. Defaults to 50.

    Returns:
        tf. Tensor: integrated gradients.
    """
    if baseline is None:
        baseline = np.zeros(img_size).astype(np.float32)
    else:
        baseline = baseline.astype(np.float32)

    img_input = img_input.astype(np.float32)
    interpolated_image = [
        baseline + (step / num_steps) * (img_input - baseline)
        for step in range(num_steps + 1)
    ]

    grads = []
    for _, img in enumerate(interpolated_image):
        img = xception.preprocess_input(img)
        img = tf.expand_dims(img, axis=0)
        grad = get_gradients(img, top_pred_idx, model)
        grads.append(grad[0])

    grads = tf.convert_to_tensor(grads, dtype=tf.float32)

    # Approximate the integral using trapezoidal rule.
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = tf.reduce_mean(grads, axis=0)

    integrated_grads = (img_input - baseline) * avg_grads
    return integrated_grads


def random_baseline_integrated_gradients(
    model: tf.keras.Model,
    img_size: Tuple,
    img_input: tf.Tensor,
    top_pred_idx: int,
    num_steps: int = 50,
    num_runs: int = 2,
) -> tf.Tensor:
    """Run Inegrated Gradients with Random Baseline.

    Args:
        model (tf.keras.Model): classfication model.
        img_size (Tuple): image size.
        img_input (tf.Tensor): input image.
        top_pred_idx (int): prediction index.
        num_steps (int, optional): number of steps. Defaults to 50.
        num_runs (int, optional): number of runs. Defaults to 2.

    Returns:
        tf.Tensor: Integrated gradients.
    """
    integrated_grads = []
    for _ in range(num_runs):
        baseline = np.random.random(img_size) * 255
        igrads = get_integrated_gradients(
            model,
            img_size,
            img_input=img_input,
            top_pred_idx=top_pred_idx,
            baseline=baseline,
            num_steps=num_steps,
        )
        integrated_grads.append(igrads)

    integrated_grads = tf.convert_to_tensor(integrated_grads)
    return tf.reduce_mean(integrated_grads, axis=0)


class GradVisualizer:
    """Plot gradients of the outputs w.r.t an input image."""

    def __init__(self, positive_channel: List = None, negative_channel: List = None):
        """Initialize Class.

        Args:
            positive_channel (List, optional): positive channel. Defaults to None.
            negative_channel (List, optional): negative channel. Defaults to None.
        """
        if positive_channel is None:
            self.positive_channel = [0, 255, 0]
        else:
            self.positive_channel = positive_channel

        if negative_channel is None:
            self.negative_channel = [255, 0, 0]
        else:
            self.negative_channel = negative_channel

    def apply_polarity(self, attributions: np.array, polarity: str) -> np.array:
        """Apply polarity.

        Args:
            attributions (np.array): gradients.
            polarity (str): positive or negative.

        Returns:
            np.array: clipped array.
        """
        if polarity == "positive":
            return np.clip(attributions, 0, 1)
        else:
            return np.clip(attributions, -1, 0)

    def apply_linear_transformation(
        self,
        attributions: np.array,
        clip_above_percentile: float = 99.9,
        clip_below_percentile: float = 70.0,
        lower_end: float = 0.2,
    ):
        """Apply linear transformation.

        Args:
            attributions (np.array): gradients.
            clip_above_percentile (float, optional): clipping ceiling.. Defaults to 99.9.
            clip_below_percentile (float, optional): clipping floor.. Defaults to 70.0.
            lower_end (float, optional): lower threshold. Defaults to 0.2.

        Returns:
            _type_: _description_
        """
        # 1. Get the thresholds
        m = self.get_thresholded_attributions(
            attributions, percentage=100 - clip_above_percentile
        )
        e = self.get_thresholded_attributions(
            attributions, percentage=100 - clip_below_percentile
        )

        # 2. Transform the attributions by a linear function f(x) = a*x + b such that
        # f(m) = 1.0 and f(e) = lower_end
        transformed_attributions = (1 - lower_end) * (np.abs(attributions) - e) / (
            m - e
        ) + lower_end

        # 3. Make sure that the sign of transformed attributions is the same as original attributions
        transformed_attributions *= np.sign(attributions)

        # 4. Only keep values that are bigger than the lower_end
        transformed_attributions *= transformed_attributions >= lower_end

        # 5. Clip values and return
        transformed_attributions = np.clip(transformed_attributions, 0.0, 1.0)
        return transformed_attributions

    def get_thresholded_attributions(
        self, attributions: np.array, percentage: float
    ) -> np.array:
        """Threshold Gradients.

        Args:
            attributions (np.array): gradients.
            percentage (float): threshold percentage.

        Returns:
            np.array: thresholded gradients.
        """
        if percentage == 100.0:
            return np.min(attributions)

        # 1. Flatten the attributions
        flatten_attr = attributions.flatten()

        # 2. Get the sum of the attributions
        total = np.sum(flatten_attr)

        # 3. Sort the attributions from largest to smallest.
        sorted_attributions = np.sort(np.abs(flatten_attr))[::-1]

        # 4. Calculate the percentage of the total sum that each attribution
        # and the values about it contribute.
        cum_sum = 100.0 * np.cumsum(sorted_attributions) / total

        # 5. Threshold the attributions by the percentage
        indices_to_consider = np.where(cum_sum >= percentage)[0][0]

        # 6. Select the desired attributions and return
        attributions = sorted_attributions[indices_to_consider]
        return attributions

    def binarize(self, attributions: np.array, threshold: float = 0.001) -> np.array:
        """Binarize gradients.

        Args:
            attributions (np.array): gradients.
            threshold (float, optional): threshold. Defaults to 0.001.

        Returns:
            np.array: boolean array.
        """
        return attributions > threshold

    def morphological_cleanup_fn(
        self, attributions: np.array, structure: np.array = np.ones((4, 4))
    ) -> np.array:
        """Morphological Cleaning.

        Args:
            attributions (np.array): gradients.
            structure (np.array, optional): structural element. Defaults to np.ones((4, 4)).

        Returns:
            np.array: cleand array.
        """
        closed = ndimage.grey_closing(attributions, structure=structure)
        opened = ndimage.grey_opening(closed, structure=structure)
        return opened

    def draw_outlines(
        self,
        attributions: np.array,
        percentage: float = 90,
        connected_component_structure: np.array = np.ones((3, 3)),
    ) -> np.array:
        """Draw Outlines.

        Args:
            attributions (np.array): gradients.
            percentage (float, optional): percentage. Defaults to 90.
            connected_component_structure (np.array, optional): structural element. Defaults to np.ones((3, 3)).

        Returns:
            np.array: outline array.
        """
        # 1. Binarize the attributions.
        attributions = self.binarize(attributions)

        # 2. Fill the gaps
        attributions = ndimage.binary_fill_holes(attributions)

        # 3. Compute connected components
        connected_components, num_comp = ndimage.measurements.label(
            attributions, structure=connected_component_structure
        )

        # 4. Sum up the attributions for each component
        total = np.sum(attributions[connected_components > 0])
        component_sums = []
        for comp in range(1, num_comp + 1):
            mask = connected_components == comp
            component_sum = np.sum(attributions[mask])
            component_sums.append((component_sum, mask))

        # 5. Compute the percentage of top components to keep
        sorted_sums_and_masks = sorted(component_sums, key=lambda x: x[0], reverse=True)
        sorted_sums = list(zip(*sorted_sums_and_masks))[0]
        cumulative_sorted_sums = np.cumsum(sorted_sums)
        cutoff_threshold = percentage * total / 100
        cutoff_idx = np.where(cumulative_sorted_sums >= cutoff_threshold)[0][0]
        if cutoff_idx > 2:
            cutoff_idx = 2

        # 6. Set the values for the kept components
        border_mask = np.zeros_like(attributions)
        for i in range(cutoff_idx + 1):
            border_mask[sorted_sums_and_masks[i][1]] = 1

        # 7. Make the mask hollow and show only the border
        eroded_mask = ndimage.binary_erosion(border_mask, iterations=1)
        border_mask[eroded_mask] = 0

        # 8. Return the outlined mask
        return border_mask

    def process_grads(
        self,
        image: np.array,
        attributions: np.array,
        polarity: str = "positive",
        clip_above_percentile: float = 99.9,
        clip_below_percentile: float = 0,
        morphological_cleanup: Boolean = False,
        structure: Tuple = np.ones((3, 3)),
        outlines: Boolean = False,
        outlines_component_percentage: float = 90,
        overlay: Boolean = True,
    ) -> np.array:
        """Process gradients.

        Args:
            image (np.array): input image.
            attributions (np.array): attributions.
            polarity (str, optional): polarity. Defaults to "positive".
            clip_above_percentile (float, optional): clip ceiling.. Defaults to 99.9.
            clip_below_percentile (float, optional): clip floor. Defaults to 0.
            morphological_cleanup (Boolean, optional): do morphological cleanup. Defaults to False.
            structure (Tuple, optional): structural element. Defaults to np.ones((3, 3)).
            outlines (Boolean, optional): draw outlines. Defaults to False.
            outlines_component_percentage (float, optional): outline percentage. Defaults to 90.
            overlay (Boolean, optional): overlay. Defaults to True.

        Raises:
            ValueError: polarity error.
            ValueError: clipping above error.
            ValueError: clipping below error.

        Returns:
            np.array: attributions.
        """
        if polarity not in ["positive", "negative"]:
            raise ValueError(
                f" Allowed polarity values: 'positive' or 'negative' but provided {polarity}"
            )
        if clip_above_percentile < 0 or clip_above_percentile > 100:
            raise ValueError("clip_above_percentile must be in [0, 100]")

        if clip_below_percentile < 0 or clip_below_percentile > 100:
            raise ValueError("clip_below_percentile must be in [0, 100]")

        # 1. Apply polarity
        if polarity == "positive":
            attributions = self.apply_polarity(attributions, polarity=polarity)
            channel = self.positive_channel
        else:
            attributions = self.apply_polarity(attributions, polarity=polarity)
            attributions = np.abs(attributions)
            channel = self.negative_channel

        # 2. Take average over the channels
        attributions = np.average(attributions, axis=2)

        # 3. Apply linear transformation to the attributions
        attributions = self.apply_linear_transformation(
            attributions,
            clip_above_percentile=clip_above_percentile,
            clip_below_percentile=clip_below_percentile,
            lower_end=0.0,
        )

        # 4. Cleanup
        if morphological_cleanup:
            attributions = self.morphological_cleanup_fn(
                attributions, structure=structure
            )
        # 5. Draw the outlines
        if outlines:
            attributions = self.draw_outlines(
                attributions, percentage=outlines_component_percentage
            )

        # 6. Expand the channel axis and convert to RGB
        attributions = np.expand_dims(attributions, 2) * channel

        # 7.Superimpose on the original image
        if overlay:
            attributions = np.clip((attributions * 0.8 + image), 0, 255)
        return attributions

    def visualize(
        self,
        image: np.array,
        gradients: np.array,
        integrated_gradients: np.array,
        output_file: str,
        polarity: str = "positive",
        clip_above_percentile: float = 99.9,
        clip_below_percentile: float = 0,
        morphological_cleanup: Boolean = False,
        structure: np.array = np.ones((3, 3)),
        outlines: Boolean = False,
        outlines_component_percentage: float = 90,
        overlay: Boolean = True,
        figsize: Tuple = (15, 8),
    ) -> None:
        """Visualize Results.

        Args:
            image (np.array): input image.
            gradients (np.array): gradients.
            integrated_gradients (np.array): integrated gradients.
            output_file (str): output filename.
            polarity (str, optional): polarity. Defaults to "positive".
            clip_above_percentile (float, optional): clip ceiling. Defaults to 99.9.
            clip_below_percentile (float, optional): clip floor. Defaults to 0.
            morphological_cleanup (Boolean, optional): do morphological cleaning. Defaults to False.
            structure (np.array, optional): structural element. Defaults to np.ones((3, 3)).
            outlines (Boolean, optional): draw outline. Defaults to False.
            outlines_component_percentage (float, optional): outline percentage. Defaults to 90.
            overlay (Boolean, optional): overlay. Defaults to True.
            figsize (Tuple, optional): figure size. Defaults to (15, 8).
        """
        # 1. Make two copies of the original image
        img1 = np.copy(image)
        img2 = np.copy(image)

        # 2. Process the normal gradients
        grads_attr = self.process_grads(
            image=img1,
            attributions=gradients,
            polarity=polarity,
            clip_above_percentile=clip_above_percentile,
            clip_below_percentile=clip_below_percentile,
            morphological_cleanup=morphological_cleanup,
            structure=structure,
            outlines=outlines,
            outlines_component_percentage=outlines_component_percentage,
            overlay=overlay,
        )

        # 3. Process the integrated gradients
        igrads_attr = self.process_grads(
            image=img2,
            attributions=integrated_gradients,
            polarity=polarity,
            clip_above_percentile=clip_above_percentile,
            clip_below_percentile=clip_below_percentile,
            morphological_cleanup=morphological_cleanup,
            structure=structure,
            outlines=outlines,
            outlines_component_percentage=outlines_component_percentage,
            overlay=overlay,
        )

        fig, ax = plt.subplots(1, 3, figsize=figsize)
        ax[0].imshow(image)
        ax[1].imshow(grads_attr.astype(np.uint8))
        ax[2].imshow(igrads_attr.astype(np.uint8))

        ax[0].set_title("Input")
        ax[1].set_title("Normal gradients")
        ax[2].set_title("Integrated gradients")
        # plt.show()
        fig.savefig(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="./images")
    main(parser.parse_args())
