import os
import numpy as np
import tensorflow as tf
import pandas as pd  # NEW

def load_and_prepare_image_tf(img_path, img_height=28, img_width=28):
    img_raw = tf.io.read_file(img_path)
    img = tf.image.decode_image(img_raw, channels=3)
    img = tf.image.resize(img, [img_height, img_width])
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    return img

def predict_image(model, img_path):
    img = load_and_prepare_image_tf(img_path)
    logits = model.predict(img, verbose=0)
    probabilities = tf.nn.softmax(logits).numpy()
    predicted_class = int(np.argmax(probabilities, axis=1)[0])
    confidence = float(np.max(probabilities))  # Highest probability

    return {
        "Image": os.path.basename(img_path),
        "Predicted Class": predicted_class,
        "Confidence": f"{confidence:.4f}"
    }

if __name__ == "__main__":
    model_path = 'best_model.keras'
    test_folder = './test'
    model = tf.keras.models.load_model(model_path)

    results = []

    for filename in os.listdir(test_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(test_folder, filename)
            result = predict_image(model, img_path)
            results.append(result)

    # Convert to DataFrame and display as table
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
