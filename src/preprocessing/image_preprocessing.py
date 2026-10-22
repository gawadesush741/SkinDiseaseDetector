import cv2
import numpy as np

image_path = "data/raw/test_skin.png"
output_path = "data/processed/test_skin_resized.png"

image = cv2.imread(image_path)

if image is None:
    print("❌ Could not open the image.")
else:
    print("✅ Original image opened!")
    print("Original size:", image.shape[1], "x", image.shape[0])

    resized_image = cv2.resize(image, (224, 224))

    print("✅ Image resized!")
    print("New size:", resized_image.shape[1], "x", resized_image.shape[0])

    normalized_image = resized_image.astype(np.float32) / 255.0

    print("✅ Image normalized!")
    print("Minimum pixel value:", normalized_image.min())
    print("Maximum pixel value:", normalized_image.max())

    cv2.imwrite(output_path, resized_image)

    print("✅ Processed image saved!")