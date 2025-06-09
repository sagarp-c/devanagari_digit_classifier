from PIL import Image, ImageEnhance, ImageOps
import os

input_base = 'training_data/numerals'
output_base = 'augmented_data/numerals'
os.makedirs(output_base, exist_ok=True)

# Define 6 different augmentation functions
def augmentations(image):
    return [
        image,  # original
        ImageEnhance.Contrast(image).enhance(1.2),
        ImageEnhance.Brightness(image).enhance(1.1),
        image.rotate(1, expand=True, fillcolor=255),
        image.rotate(-1, expand=True, fillcolor=255),
        ImageOps.mirror(image),
    ]

# Process each class (0 to 9)
for class_name in sorted(os.listdir(input_base)):
    class_path = os.path.join(input_base, class_name)
    output_class_path = os.path.join(output_base, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    if not os.path.isdir(class_path):
        continue

    counter = 0
    for file in sorted(os.listdir(class_path)):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(class_path, file)
            try:
                img = Image.open(img_path).convert('L')
            except Exception as e:
                print(f"❌ Failed to open image: {img_path}. Error: {e}")
                continue

            base_name = os.path.splitext(file)[0]
            for i, aug_img in enumerate(augmentations(img)):
                aug_filename = f"{base_name}_aug{i}.png"
                aug_img_path = os.path.join(output_class_path, aug_filename)
                aug_img.save(aug_img_path)
                counter += 1

    print(f"✅ Class {class_name}: {counter} augmented images saved to {output_class_path}")

print("\n🎉 All augmentations complete!")
