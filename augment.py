from PIL import Image, ImageEnhance, ImageOps
import os
import numpy as np

input_base = 'training_data/numerals'
output_base = 'augmented_data/numerals'
os.makedirs(output_base, exist_ok=True)

def add_noise(img, amount=30):
    # Add random pixel noise
    arr = np.array(img).astype(np.int16)
    noise = np.random.randint(-amount, amount+1, arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_arr)

def augmentations(image):
    # Small rotation angles to preserve meaning
    rotations = [0, 1, -1, 3, -3]
    # Small shear factors (affine transform)
    shears = [0, 0.1, -0.1]

    augmented_images = []

    # Original and contrast/brightness variants
    augmented_images.append(image)
    augmented_images.append(ImageEnhance.Contrast(image).enhance(1.2))
    augmented_images.append(ImageEnhance.Brightness(image).enhance(1.1))

    # Rotations only small angles
    for angle in rotations[1:]:
        rotated = image.rotate(angle, fillcolor=255, expand=False)
        augmented_images.append(rotated)

    # Small shears via affine transform
    for shear in shears[1:]:
        width, height = image.size
        m = shear
        xshift = abs(m) * height
        new_width = width + int(round(xshift))
        affine_img = image.transform(
            (new_width, height),
            Image.AFFINE,
            (1, m, -xshift if m > 0 else 0, 0, 1, 0),
            fillcolor=255
        )
        # Resize back to original size
        affine_img = affine_img.resize((width, height), Image.BILINEAR)
        augmented_images.append(affine_img)

    # Add noise images
    augmented_images.append(add_noise(image, amount=30))
    augmented_images.append(add_noise(image, amount=60))

    return augmented_images

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
