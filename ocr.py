import cv2
import pytesseract
from collections import defaultdict
import random
import re

# Path to the Tesseract executable (adjust for your system)
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"  # Linux/Mac
# For Windows: r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Specify Hebrew as the language
custom_config = r'--oem 3 --psm 6 -l heb'

# Load the image using OpenCV
image_path = "/home/yakir/sticker_ws/1.jpeg"  # Replace with your image path
image = cv2.imread(image_path)

# Perform OCR to get word-level bounding boxes and text
h, w, _ = image.shape  # Image dimensions
data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)

# Threshold for grouping words into lines
line_threshold = 20  # Adjust based on the image

# Group words by line
lines = defaultdict(list)  # Dictionary to hold words grouped by y-coordinates

for i in range(len(data['text'])):
    if int(data['conf'][i]) > 0:  # Filter out weak detections
        # Extract word details
        word = data['text'][i]
        if len(word) <= 1  or 'וו' in word or '||' in word:   # Filter words with numbers or only one char
            continue
      
        x, y, width, height = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        
        # Determine the line index (using y-coordinate with threshold)
        line_key = y // line_threshold
        lines[line_key].append((word, x, y, width, height))

# Sort lines by their y-coordinates
sorted_lines = sorted(lines.items(), key=lambda item: item[0])

# Create the list of lists for words grouped by lines
words_by_lines = [[word[0] for word in line[1]] for line in sorted_lines]

# Generate unique colors for each line
unique_colors = {
    line_key: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for line_key in lines.keys()
}

# Draw bounding boxes and overlay words on the image
for line_key, line_words in sorted_lines:
    color = unique_colors[line_key]  # Get unique color for this line
    for word, x, y, width, height in line_words:
        # Draw bounding box with line-specific color
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 3)


# Define the scale factors
scale_x = 0.5  # Resize width to 50%
scale_y = 0.5  # Resize height to 50%
        
image = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)


# Optionally save the result
cv2.imwrite("output_with_colored_lines.jpg", image)

# Print the words grouped by lines
for i, line in enumerate(words_by_lines):
    print(f"Line {i + 1}: {line}")
# Save or display the annotated image
cv2.imshow("Annotated Image with Colored Lines", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
