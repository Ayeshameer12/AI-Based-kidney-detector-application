"""
gradcam.py
----------
Grad-CAM implementation: shows which region of the image the CNN
focused on to make its Normal/Abnormal prediction. This gives the
"detected area highlighted" behavior from the proposal's flowchart
without needing bounding-box-annotated training data.
"""

import numpy as np
import tensorflow as tf
import cv2


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    img_array: preprocessed image batch, shape (1, H, W, 3)
    model: trained Keras model
    last_conv_layer_name: name of the last convolutional layer to inspect
    Returns: 2D numpy heatmap, values in [0, 1]
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # Binary sigmoid output — use it directly as the "class score"
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_img_path, heatmap, output_path, alpha=0.4):
    """
    Overlays the Grad-CAM heatmap on top of the original image and saves it.
    """
    img = cv2.imread(original_img_path)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlaid = cv2.addWeighted(heatmap_colored, alpha, img, 1 - alpha, 0)
    cv2.imwrite(output_path, overlaid)
