import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd  # For saving train log csv

def plot_training_history(history):
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_performance.png')
    print("Training graph saved as 'training_performance.png'")
    plt.close()

def save_train_log(history, filename='train_log.csv'):
    df = pd.DataFrame(history.history)
    df.index += 1  # Epochs starting from 1
    df.to_csv(filename, index_label='Epoch')
    print(f"Training log saved as '{filename}'")

def evaluate_and_plot(model, data_generator, model_name):
    print(f"Evaluating {model_name}...")

    results = model.evaluate(data_generator, verbose=0)
    loss = results[0]
    accuracy = results[1]

    plt.figure(figsize=(6,4))
    plt.bar(['Loss', 'Accuracy'], [loss, accuracy], color=['red', 'green'])
    plt.ylim([0, 1.1])
    plt.title(f'{model_name} Evaluation')
    plt.ylabel('Value')
    for i, v in enumerate([loss, accuracy]):
        plt.text(i, v + 0.05, f"{v:.4f}", ha='center')
    filename = f"{model_name.lower().replace(' ', '_')}_performance.png"
    plt.savefig(filename)
    print(f"{model_name} performance graph saved as '{filename}'")
    plt.close()

def detailed_evaluation_and_plot(model, data_generator, model_name):
    print(f"Running detailed evaluation for {model_name}...")

    y_true = data_generator.classes
    steps = data_generator.samples // data_generator.batch_size
    if data_generator.samples % data_generator.batch_size != 0:
        steps += 1

    y_pred_probs = model.predict(data_generator, steps=steps, verbose=0)
    y_pred = tf.argmax(y_pred_probs, axis=1).numpy()

    report = classification_report(y_true, y_pred, output_dict=True)
    print(f"\nClassification Report for {model_name}:\n", classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.close()
    print(f"{model_name} confusion matrix saved.")

    labels = list(report.keys())[:-3]  # skip accuracy, macro avg, weighted avg
    precision = [report[label]['precision'] for label in labels]
    recall = [report[label]['recall'] for label in labels]
    f1 = [report[label]['f1-score'] for label in labels]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(10,6))
    plt.bar(x - width, precision, width=width, label='Precision')
    plt.bar(x, recall, width=width, label='Recall')
    plt.bar(x + width, f1, width=width, label='F1-Score')

    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.title(f'{model_name} Per-Class Precision, Recall, F1-Score')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_detailed_metrics.png')
    plt.close()
    print(f"{model_name} detailed metrics plot saved.")

def main():
    data_dir = './training_data/numerals'
    batch_size = 32
    img_height = 64
    img_width = 64
    epochs = 20

    best_model_path = 'best_model_checkpoint.keras'
    last_model_path = 'final_model.keras'

    # Only rescaling, no augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='sparse',
        subset='training',
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation'
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3),
                               kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                               kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu',
                              kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(10)  # 10 classes
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    callbacks = [
        ModelCheckpoint(filepath=best_model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=callbacks
    )

    model.save(last_model_path)
    print(f"Final model saved as '{last_model_path}'")
    print(f"Best model saved automatically as '{best_model_path}'")

    plot_training_history(history)
    save_train_log(history)

    best_model = tf.keras.models.load_model(best_model_path)
    evaluate_and_plot(best_model, validation_generator, "Best Model")
    detailed_evaluation_and_plot(best_model, validation_generator, "Best Model")

    last_model = tf.keras.models.load_model(last_model_path)
    evaluate_and_plot(last_model, validation_generator, "Last Model")
    detailed_evaluation_and_plot(last_model, validation_generator, "Last Model")

if __name__ == "__main__":
    main()
