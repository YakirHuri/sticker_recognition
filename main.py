# from pyzbar.pyzbar import decode
# import cv2
# import numpy as np

# def detect_barcodes_with_pyzbar(image_path):
#     # Read the image
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Decode the barcodes
#     barcodes = decode(gray)
#     print(f"Number of barcodes detected: {len(barcodes)}")

#     for barcode in barcodes:
#         # Extract the bounding box coordinates
#         points = barcode.polygon
#         if len(points) == 4:
#             # Draw a polygon around the barcode
#             pts = np.array([(point.x, point.y) for point in points], dtype=np.int32)
#             cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

#             # Get the bounding rectangle for a simple rectangle outline
#             x, y, w, h = cv2.boundingRect(pts)
#             cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
#         # Get the barcode data
#         barcode_data = barcode.data.decode("utf-8")
#         barcode_type = barcode.type

#         # Display the barcode data on the image
#         x, y = points[0].x, points[0].y
#         cv2.putText(image, f"{barcode_type}: {barcode_data}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Resize the image for better display
#     scale_x, scale_y = 1.3, 1.3
#     resized_image = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

#     # Show the result
#     cv2.imshow("Detected Barcodes", resized_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Path to the image containing barcodes
# image_path = '/home/yakir/sticker_ws/3.jpeg'
# detect_barcodes_with_pyzbar(image_path)


from pyzbar.pyzbar import decode
from PIL import Image

# Load the image
image_path = "/home/yakir/sticker_ws/1.jpeg"  # Path to your uploaded image
image = Image.open(image_path)

# Decode the barcode
decoded_objects = decode(image)

# Print the results
for obj in decoded_objects:
    print("Type:", obj.type)
    print("Data:", obj.data.decode("utf-8"))
