
import os
import numpy as np
import tensorflow as tf

def load_and_prepare_image_tf(img_path, img_height=28, img_width=28):
    """
    Loads and preprocesses an image using TensorFlow I/O.
    Returns a normalized 4D tensor: (1, height, width, 3)
    """
    # Load image as byte string
    img_raw = tf.io.read_file(img_path)
    
    # Decode JPEG/PNG
    img = tf.image.decode_image(img_raw, channels=3)  # Automatically handles PNG/JPEG
    
    # Resize to model input size
    img = tf.image.resize(img, [img_height, img_width])
    
    # Normalize to [0, 1]
    img = img / 255.0
    
    # Add batch dimension: (1, height, width, 3)
    img = tf.expand_dims(img, axis=0)
    
    return img

def predict_image(model, img_path):
    """
    Runs prediction on a single image and prints the result.
    """
    img = load_and_prepare_image_tf(img_path)

    logits = model.predict(img)
    probabilities = tf.nn.softmax(logits).numpy()
    predicted_class = np.argmax(probabilities, axis=1)[0]

    print(f"Image: {os.path.basename(img_path)}")
    print(f"Predicted class: {predicted_class}")
    print(f"Class probabilities: {probabilities}")
    print("-" * 40)

if __name__ == "__main__":
    model_path = 'best_model.keras'
    test_folder = './test'

    # Load model once
    model = tf.keras.models.load_model(model_path)
    print("Model input shape:", model.input_shape)

    # Iterate over all images in test folder
    for filename in os.listdir(test_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(test_folder, filename)
            predict_image(model, img_path)

