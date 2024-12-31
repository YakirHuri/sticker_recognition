import cv2
import pytesseract
from collections import defaultdict
import random

class OCRProcessor:
    def __init__(self, tesseract_cmd, image_path, output_path, lang='heb'):
        self.image_path = image_path
        self.output_path = output_path
        self.custom_config = f'--oem 3 --psm 6 -l {lang}'
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        self.wanted_fieldes = ['מקרה','אלמוני','זמני', 'ת.קבלה', 'מלרד']
        
  

    def align_by_white_area(self, image, direction_angle):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = image[y:y + h, x:x + w]
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[-1]
        if angle < -45:
            angle += 90

        angle+=direction_angle
        # Calculate the bounding box dimensions after rotation
        rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        new_width = int((h * sin) + (w * cos))
        new_height = int((h * cos) + (w * sin))

        # Adjust the rotation matrix to account for the new bounding dimensions
        rotation_matrix[0, 2] += (new_width / 2) - (w / 2)
        rotation_matrix[1, 2] += (new_height / 2) - (h / 2)

        aligned_image = cv2.warpAffine(cropped_image, rotation_matrix, (new_width, new_height))
        return aligned_image

    def detect_keywords(self, image):
        # Get OCR data
        data = pytesseract.image_to_data(image, config=self.custom_config, output_type=pytesseract.Output.DICT)
        
        # Iterate through each detected word
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:  # Confidence level > 0
                word = data['text'][i]
                if word == 'אלמוני':               
                  
                    return True, image
     
        return False, image
    
    def procces(self, image):
        # Get OCR data
        data = pytesseract.image_to_data(image, config=self.custom_config, output_type=pytesseract.Output.DICT)
        
        # Iterate through each detected word
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:  # Confidence level > 0
                word = data['text'][i]
                print(f'word is {word}') 
                # Get bounding box coordinates
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                
                # Draw rectangle on the image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box with thickness 2
                
                # Write the word above the rectangle
                text_x = x
                text_y = y - 10  # Adjust to place text above the box
                cv2.putText(image, word, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)  # Blue text with thickness 1
        
        # Save and display the image
        cv2.imwrite('/home/yakir/sticker_ws/test/detection.jpg', image)
        cv2.imshow("Detected Words", image)
        cv2.waitKey(0)
       
    
    def perform_ocr(self):
        image = cv2.imread(self.image_path)
        print('align iamge:')

        
        res1, img1 = self.detect_keywords(self.align_by_white_area(image,90))
        res2, img2 = self.detect_keywords(self.align_by_white_area(image,-90))
        res3, img3 = self.detect_keywords(self.align_by_white_area(image,180))
        res4, img4 = self.detect_keywords(self.align_by_white_area(image,-180))

        print(f" res1 {res1} res2 {res2} res3 {res3} res4 {res4}")

        if res1: 
            self.procces(img1)       
        if res2:        
            self.procces(img2)
        if res3:        
            self.procces(img3)  
        if res4:        
            self.procces(img4)  
                      
        # cv2.imshow("aligned_image_pos_90", aligned_image_pos_90)
        # cv2.imshow("aligned_image_neg_90", aligned_image_neg_90)
        # cv2.imshow("aligned_image_pos_180", aligned_image_pos_180)
        # cv2.imshow("aligned_image_neg_180", aligned_image_neg_180)



        # h, w, _ = aligned_image.shape
        # data = pytesseract.image_to_data(aligned_image, config=self.custom_config, output_type=pytesseract.Output.DICT)

        # line_threshold = 20
        # lines = defaultdict(list)

        # for i in range(len(data['text'])):
        #     if int(data['conf'][i]) > 0:
        #         word = data['text'][i]
        #         if len(word) <= 1 or 'וו' in word or '||' in word:
        #             continue
        #         x, y, width, height = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        #         line_key = y // line_threshold
        #         lines[line_key].append((word, x, y, width, height))

        # sorted_lines = sorted(lines.items(), key=lambda item: item[0])
        # words_by_lines = [[word[0] for word in line[1]] for line in sorted_lines]

        # unique_colors = {
        #     line_key: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        #     for line_key in lines.keys()
        # }

        # for line_key, line_words in sorted_lines:
        #     color = unique_colors[line_key]
        #     for word, x, y, width, height in line_words:
        #         cv2.rectangle(aligned_image, (x, y), (x + width, y + height), color, 3)

        # scale_x, scale_y = 0.5, 0.5
        # resized_image = cv2.resize(aligned_image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
        # cv2.imwrite(self.output_path, resized_image)

        # for i, line in enumerate(words_by_lines):
        #     print(f"Line {i + 1}: {line}")

        

        # cv2.imshow("Annotated Image with Colored Lines", resized_image)
# Example usage
if __name__ == "__main__":
    tesseract_cmd = r"/usr/bin/tesseract"  # Adjust based on your system
    image_path = "/home/yakir/sticker_ws/output/14.jpg"
    output_path = "/home/yakir/sticker_ws/test/output_with_colored_lines.jpg"

    ocr_processor = OCRProcessor(tesseract_cmd, image_path, output_path)
    ocr_processor.perform_ocr()

# 13 good
# 3 good
# 4 good
# 6 good
# 7 good
# 8 good
# 10 good
# 11 good
# 12 good
# 13 good
# 13 good
# 14 good
# 20 good





# 1 bad
# 2 bad
# 5 bad
# 9 bad
# 15 bad
# 16 bad
# 17 bad
# 18
# 19
# 20
