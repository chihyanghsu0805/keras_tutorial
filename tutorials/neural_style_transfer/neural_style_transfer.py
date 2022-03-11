"""This code reimplements the tutorial on https://www.tensorflow.org/tutorials/generative/style_transfer."""

from __future__ import absolute_import, print_function

import argparse
import os
import time
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub

os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "COMPRESSED"
mpl.rcParams["figure.figsize"] = (12, 12)
mpl.rcParams["axes.grid"] = False


def load_img(path_to_img: str) -> tf.Tensor:
    """Load images.

    Args:
        path_to_img (str): image path.

    Returns:
        tf.Tensor: image as tensor.
    """
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def imshow(image: tf.Tensor, title: str = None) -> None:
    """Show image.

    Args:
        image (tf.Tensor): image as a tensor.
        title (str, optional): image title. Defaults to None.
    """
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


def vgg_layers(layer_names: str) -> tf.keras.Model:
    """Create a vgg model that returns a list of intermediate output values.

    Args:
        layer_names (str): name of layer.

    Returns:
        tf.keras.Model: model with list of intermediate layer outputs.
    """
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


def gram_matrix(input_tensor: tf.Tensor) -> tf.Tensor:
    """Compute Gram Matrix.

    Args:
        input_tensor (tf.Tensor): input tensors.

    Returns:
        tf.Tensor: gram matrix.
    """
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)


class StyleContentModel(tf.keras.models.Model):
    """Model for style and content."""

    def __init__(self, style_layers: List, content_layers: List):
        """Initialize class.

        Args:
            style_layers (List): list of layer names.
            content_layers (List): list of layer names.
        """
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs: tf.Tensor) -> Dict:
        """Call class.

        Args:
            inputs (tf.Tensor): input tensor, Expects float input in [0,1].

        Returns:
            Dict: dict with content/style layers mapped with corresponding outputs.
        """
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (
            outputs[: self.num_style_layers],
            outputs[self.num_style_layers :],
        )

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {
            content_name: value
            for content_name, value in zip(self.content_layers, content_outputs)
        }

        style_dict = {
            style_name: value
            for style_name, value in zip(self.style_layers, style_outputs)
        }

        return {"content": content_dict, "style": style_dict}


def clip_0_1(image: tf.Tensor) -> tf.Tensor:
    """Clip image intensity to [0, 1].

    Args:
        image (tf.Tensor): original image.

    Returns:
        tf.Tensor: clipped image.
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def high_pass_x_y(image: tf.Tensor) -> tf.Tensor:
    """Apple high pass filter.

    Args:
        image (tf.Tensor): input image.

    Returns:
        tf.Tensor: filtered image.
    """
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

    return x_var, y_var


def style_content_loss(
    outputs: Dict,
    style_targets: List,
    style_weight: float,
    content_targets: List,
    content_weight: float,
) -> float:
    """Compute style content loss.

    Args:
        outputs (Dict): model outputs.
        style_targets (List): style layers.
        style_weight (float): style weight.
        content_targets (List): content layers.
        content_weight (weight): content weight.

    Returns:
        float: loss
    """
    style_outputs = outputs["style"]
    content_outputs = outputs["content"]
    style_loss = tf.add_n(
        [
            tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
            for name in style_outputs.keys()
        ]
    )
    style_loss *= style_weight / len(style_outputs)

    content_loss = tf.add_n(
        [
            tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
            for name in content_outputs.keys()
        ]
    )
    content_loss *= content_weight / len(content_outputs)
    loss = style_loss + content_loss
    return loss


@tf.function()
def train_step(
    extractor: tf.keras.Model,
    image: tf.Tensor,
    style_targets: List,
    style_weight: float,
    content_targets: List,
    content_weight: float,
    opt: tf.optimizers,
):
    """Run one training step.

    Args:
        extractor (tf.keras.Model): model to extract features from.
        image (tf.Tensor): input image.
        style_targets (List): layers to extract style from.
        style_weight (float): weights of style to loss.
        content_targets (List): layers to extract content from.
        content_weight (float): weights of content to loss.
        opt (tf.optimizer): model optimizer.
    """
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(
            outputs, style_targets, style_weight, content_targets, content_weight
        )

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


@tf.function()
def train_step_weighted(
    extractor: tf.keras.Model,
    image: tf.Tensor,
    style_targets: List,
    style_weight: float,
    content_targets: List,
    content_weight: float,
    opt: tf.optimizers,
    total_variation_weight: float,
):
    """Run one training step.

    Args:
        extractor (tf.keras.Model): model to extract features from.
        image (tf.Tensor): input image.
        style_targets (List): layers to extract style from.
        style_weight (float): weights of style to loss.
        content_targets (List): layers to extract content from.
        content_weight (float): weights of content to loss.
        opt (tf.optimizers): model optimizer.
        total_variation_weight (float): variation weight.
    """
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(
            outputs, style_targets, style_weight, content_targets, content_weight
        )
        loss += total_variation_weight * tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


def main(args: argparse.Namespace) -> None:
    """Run Neural Style Transfer.

    Args:
        args (argparse.Namespace): input arguments.
    """
    # Load Content and Style Images
    content_path = tf.keras.utils.get_file(
        "YellowLabradorLooking_new.jpg",
        "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg",
    )

    style_path = tf.keras.utils.get_file(
        "kandinsky5.jpg",
        "https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg",
    )

    content_image = load_img(content_path)
    style_image = load_img(style_path)

    os.makedirs(args.image_dir, exist_ok=True)

    fig = plt.figure()
    imshow(content_image, "Content Image")
    output_file = os.path.join(args.image_dir, "content_image.jpeg")
    fig.savefig(output_file)

    fig = plt.figure()
    imshow(style_image, "Style Image")
    output_file = os.path.join(args.image_dir, "style_image.jpeg")
    fig.savefig(output_file)

    # Load Model
    hub_model = hub.load(
        "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
    )
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    fig = plt.figure()
    imshow(stylized_image, "Stylized Image")
    output_file = os.path.join(args.image_dir, "stylized_image.jpeg")
    fig.savefig(output_file)

    # Load VGG19 and Test
    x = tf.keras.applications.vgg19.preprocess_input(content_image * 255)
    x = tf.image.resize(x, (224, 224))
    vgg = tf.keras.applications.VGG19(include_top=True, weights="imagenet")

    prediction_probabilities = vgg(x)
    print(prediction_probabilities.shape)

    predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(
        prediction_probabilities.numpy()
    )[0]
    print([(class_name, prob) for (_, class_name, prob) in predicted_top_5])

    # Use VGG19 as extractor, include_top=False excludes the classification head
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")

    print("Layer Names:")
    for layer in vgg.layers:
        print(layer.name)

    # Assign layers to extract from
    content_layers = ["block5_conv2"]

    style_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]

    # Extract styles
    style_extractor = vgg_layers(style_layers)
    style_outputs = style_extractor(style_image * 255)

    # Look at the statistics of each layer"s output
    for name, output in zip(style_layers, style_outputs):
        print(name)
        print("  shape: ", output.numpy().shape)
        print("  min: ", output.numpy().min())
        print("  max: ", output.numpy().max())
        print("  mean: ", output.numpy().mean())
        print()

    extractor = StyleContentModel(style_layers, content_layers)

    results = extractor(tf.constant(content_image))

    print("Styles:")
    for name, output in sorted(results["style"].items()):
        print("  ", name)
        print("    shape: ", output.numpy().shape)
        print("    min: ", output.numpy().min())
        print("    max: ", output.numpy().max())
        print("    mean: ", output.numpy().mean())
        print()

    print("Contents:")
    for name, output in sorted(results["content"].items()):
        print("  ", name)
        print("    shape: ", output.numpy().shape)
        print("    min: ", output.numpy().min())
        print("    max: ", output.numpy().max())
        print("    mean: ", output.numpy().mean())

    # Run Gradient Descent
    style_targets = extractor(style_image)["style"]
    content_targets = extractor(content_image)["content"]
    image = tf.Variable(content_image)

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    style_weight = 1e-2
    content_weight = 1e4

    train_step(
        extractor,
        image,
        style_targets,
        style_weight,
        content_targets,
        content_weight,
        opt,
    )
    train_step(
        extractor,
        image,
        style_targets,
        style_weight,
        content_targets,
        content_weight,
        opt,
    )
    train_step(
        extractor,
        image,
        style_targets,
        style_weight,
        content_targets,
        content_weight,
        opt,
    )

    fig = plt.figure()
    imshow(image, "Test Image")
    output_file = os.path.join(args.image_dir, "test_image.jpeg")
    fig.savefig(output_file)

    # Run Longer
    start = time.time()

    epochs = 10
    steps_per_epoch = 100

    step = 0
    fig = plt.figure()
    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            step += 1
            train_step(
                extractor,
                image,
                style_targets,
                style_weight,
                content_targets,
                content_weight,
                opt,
            )
            print(".", end="", flush=True)

        print("Train step: {}".format(step))

    imshow(image, "Trained Image")
    output_file = os.path.join(args.image_dir, "train_image.jpeg")
    fig.savefig(output_file)

    end = time.time()
    print("Total time: {:.1f}".format(end - start))

    # Total Variation Loss
    x_deltas, y_deltas = high_pass_x_y(content_image)

    fig = plt.figure()
    plt.subplot(2, 2, 1)
    imshow(clip_0_1(2 * y_deltas + 0.5), "Horizontal Deltas: Original")

    plt.subplot(2, 2, 2)
    imshow(clip_0_1(2 * x_deltas + 0.5), "Vertical Deltas: Original")

    x_deltas, y_deltas = high_pass_x_y(image)

    plt.subplot(2, 2, 3)
    imshow(clip_0_1(2 * y_deltas + 0.5), "Horizontal Deltas: Styled")

    plt.subplot(2, 2, 4)
    imshow(clip_0_1(2 * x_deltas + 0.5), "Vertical Deltas: Styled")

    output_file = os.path.join(args.image_dir, "delta_image.jpeg")
    fig.savefig(output_file)

    fig = plt.figure()

    sobel = tf.image.sobel_edges(content_image)
    plt.subplot(1, 2, 1)
    imshow(clip_0_1(sobel[..., 0] / 4 + 0.5), "Horizontal Sobel-edges")
    plt.subplot(1, 2, 2)
    imshow(clip_0_1(sobel[..., 1] / 4 + 0.5), "Vertical Sobel-edges")

    output_file = os.path.join(args.image_dir, "sobel_image.jpeg")
    fig.savefig(output_file)

    print(tf.image.total_variation(image).numpy())

    # Weight Total Variation
    total_variation_weight = 30

    image = tf.Variable(content_image)

    start = time.time()

    epochs = 10
    steps_per_epoch = 100

    step = 0
    fig = plt.figure()
    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            step += 1
            train_step_weighted(
                extractor,
                image,
                style_targets,
                style_weight,
                content_targets,
                content_weight,
                opt,
                total_variation_weight,
            )
            print(".", end="", flush=True)

        print("Train step: {}".format(step))

    imshow(image, "Train Weighted Image")
    output_file = os.path.join(args.image_dir, "train_weighted_image.jpeg")
    fig.savefig(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="./images")
    main(parser.parse_args())
