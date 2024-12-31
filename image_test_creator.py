import cv2
import numpy as np

def transform_image(input_path, output_path, scale, angle):
    # Load the image
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {input_path}")

    # Get dimensions of the original image
    h, w = image.shape[:2]

    # Scale the image
    scaled_width = int(w * scale)
    scaled_height = int(h * scale)
    scaled_image = cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)

    # Calculate the new canvas size to ensure the entire rotated image fits
    diagonal = int(np.sqrt(scaled_width**2 + scaled_height**2))
    canvas_size = (diagonal, diagonal)

    # Create a black background of the new canvas size
    black_background = np.zeros((diagonal, diagonal, 3), dtype=np.uint8)

    # Compute the center offset for placing the scaled image onto the black background
    offset_x = (diagonal - scaled_width) // 2
    offset_y = (diagonal - scaled_height) // 2
    black_background[offset_y:offset_y+scaled_height, offset_x:offset_x+scaled_width] = scaled_image

    # Rotate the image with the black background
    center = (diagonal // 2, diagonal // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(black_background, rotation_matrix, canvas_size)

    # Save the result
    cv2.imwrite(output_path, rotated_image)

# Example usage
input_image_path = '/home/yakir/sticker_ws/test/7.jpeg'
output_image_path = '/home/yakir/sticker_ws/output/transformed_image.jpg'

# Apply scaling (e.g., 1.5x) and rotation (e.g., 45 degrees)
transform_image(input_image_path, output_image_path, scale=1.2, angle=17)

print(f"Transformed image saved at {output_image_path}")
