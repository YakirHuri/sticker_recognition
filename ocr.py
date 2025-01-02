import cv2
import pytesseract
from collections import defaultdict
import random
import os
import numpy as np
from dataclasses import dataclass

# הפרטים שצריך מתמונה: 
# - שם מלא
# - תעודת זהות
# - מספר מקרה
# - תאריך (התאריך בו צולמה המדבקה, היום בו רץ האלגוריתם על התמונה)
#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
@dataclass
class WordInfo:
    text: str
    rect: tuple  # (x, y, w, h)
    pixel_center: tuple  # (center_x, center_y)
    conf: int  # Confidence level
    def __str__(self):
        return f"Text: '{self.text}', Rect: {self.rect}, Pixel Center: {self.pixel_center}, Confidence: {self.conf}"
    
class OCRProcessor:
    def __init__(self ):
        
        self.custom_config = f'--oem 3 --psm 6 -l heb'
        pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"
        
        
    def crop_by_color(self, bgr_image, bgr_color, threshold=40):
        if image is None:
            raise ValueError("Input image is None. Ensure the image path is correct and accessible.")

        
        if not isinstance(image, np.ndarray):
            raise TypeError("Input is not a valid NumPy array.")     

        # Calculate Euclidean distance in bgr color space
        bgr_distance = np.sqrt(np.sum((bgr_image - bgr_color) ** 2, axis=2))

        # Create a binary mask where distance is less than the threshold
        mask = (bgr_distance < threshold).astype(np.uint8) * 255

        # Find contours to identify the largest connected component
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("No regions found with the specified color.")

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Compute the bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the original image
        cropped_image = bgr_image[y:y + h, x:x + w]
        
        # cv2.imshow('mask',mask)
        # cv2.imshow('cropped_image',cropped_image)
        # cv2.waitKey(0)
        
        
        return cropped_image       
    
    
    def process(self, image, delta=25):
        # Create a copy of the image for debugging
        debug_image = image.copy()
       
        # Get OCR data
        data = pytesseract.image_to_data(image, config=self.custom_config, output_type=pytesseract.Output.DICT)

        # List to store WordInfo objects
        word_info_list = []

        # Iterate through each detected word
        for i in range(len(data['text'])):
            if int(data['conf'][i]) >= 0:  
                word = data['text'][i]
                # if word == '|':
                #     continue
                
                conf = int(data['conf'][i])

                # Get bounding box coordinates
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

                # Calculate the center of the bounding box
                center_x = x + w // 2
                center_y = y + h // 2

                # Expand the width and height of the bounding box by delta
                new_w = w + delta
                new_h = h + delta

                # Adjust x and y to shift the bounding box and keep the center
                new_x = center_x - new_w // 2
                new_y = center_y - new_h // 2

                # Store the word details in WordInfo with expanded bounding box
                word_info = WordInfo(
                    text=word,
                    rect=(new_x, new_y, new_w, new_h),
                    pixel_center=(center_x, center_y),
                    conf=conf
                )
                word_info_list.append(word_info)

                # Draw the expanded rectangle on the image
                cv2.rectangle(debug_image, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)  # Green box with thickness 2

                # Write the word above the rectangle
                text_x = new_x
                text_y = new_y - 10  # Adjust to place text above the box

        # Save and display the image (optional)
        cv2.imwrite('/home/yakir/sticker_ws/new/output/out.png', debug_image)
        cv2.imshow("Detected Words", debug_image)
        cv2.waitKey(0)

        # Return the list of WordInfo objects
        return word_info_list


    
    def detect_name(self, words_info, y_threshold=10):
        full_name = []  # List to store the detected full name
        
        for index, word in enumerate(words_info):
            if 'שם' in word.text:           

                # Check subsequent words for the same y-range
                for i in range(index + 1, len(words_info)):
                    next_word = words_info[i]
                    
                    # If the y-coordinate is close enough, consider it part of the same name
                    if abs(next_word.rect[1] - word.rect[1]) <= y_threshold:
                        full_name.append(next_word.text)
                        # print(f"Added '{next_word.text}' to full_name.")
                    else:
                        # If the y-coordinate is too far apart, stop checking further
                        break
                
                # Print the full name once detected
                return ' '.join(full_name)  # Return the full name once it's detected

        return ''


    def detect_person_id(self, words_info, image):
        
        max_length = 0  # To track the maximum length of number sequence
        detected_id = ''  # To store the ID with the longest sequence of numbers
        image_height, image_width = image.shape[:2]  # Get image dimensions
        
        detected_ids = []
        for index, word in enumerate(words_info):
            # Check if the word contains 'מ.ז' or 'מז'
            if 'מ.ז' in word.text or 'מז' in word.text:
                # Check only the next word, if exists
                next_word = words_info[index+1]   
                            
                # Extract the bounding box coordinates of the next word
                x, y, w, h = next_word.rect

                # Ensure the bounding box stays within the image bounds
                x = max(0, x)  # Clamp x to be at least 0
                y = max(0, y)  # Clamp y to be at least 0
                w = min(w, image_width - x)  # Ensure the width does not go beyond the image
                h = min(h, image_height - y)  # Ensure the height does not go beyond the image


                # Crop the image using the bounding box of the next word
                cropped_image = image[y:y + h, x:x + w]
                # Convert to grayscale
                gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

                # Apply binary thresholding
                _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
                
                # Use pytesseract to extract text from the cropped image
                cropped_text = pytesseract.image_to_string(binary_image, config='--psm 6')
                # Skipping: Contains letters
                if any(c.isalpha() for c in cropped_text):
                    continue
                # print(f'index: {index}, Cropped text: {cropped_text}')
                
                # # Display the cropped image (optional for debugging)
                # cv2.imshow(f'Index {index}', cropped_image)
                # cv2.imshow(f'binary_image', binary_image)

                # cv2.waitKey(0)

                detected_ids.append(cropped_text)                   
                   
        
        
        if len(detected_ids) > 0:
            # Find the ID with the maximum number of numeric characters
            largest_id = max(detected_ids, key=lambda x: sum(c.isdigit() for c in x))
            
            # Create a string containing only the numeric characters from the largest ID
            numeric_id = ''.join(c for c in largest_id if c.isdigit())
            
            return numeric_id
            
        # Return the detected ID with the longest number sequence
        return detected_id

               
    
    def detect_case_number(self, words_info, image):
        
        res = self.detect_case_number_on_large_sticker(words_info, image)
        
        if res != '':
            return res

        max_length = 0  # To track the maximum length of number sequence
        detected_id = ''  # To store the ID with the longest sequence of numbers
        image_height, image_width = image.shape[:2]  # Get image dimensions
        
        detected_ids = []
        for index, word in enumerate(words_info):
            # Check if the word contains 'מ.ז' or 'מז'
            if 'מקרה' in word.text  and not any(c.isdigit() for c in word.text):
                # Check only the next word, if exists
                next_word = words_info[index+1]   
                            
                # Extract the bounding box coordinates of the next word
                x, y, w, h = next_word.rect

                # Ensure the bounding box stays within the image bounds
                x = max(0, x)  # Clamp x to be at least 0
                y = max(0, y)  # Clamp y to be at least 0
                w = min(w, image_width - x)  # Ensure the width does not go beyond the image
                h = min(h, image_height - y)  # Ensure the height does not go beyond the image


                # Crop the image using the bounding box of the next word
                cropped_image = image[y:y + h, x:x + w]
                # Convert to grayscale
                gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

                # Apply binary thresholding
                _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
                
                # Use pytesseract to extract text from the cropped image
                cropped_text = pytesseract.image_to_string(binary_image, config='--psm 6')
                # Skipping: Contains letters
                if any(c.isalpha() for c in cropped_text):
                    continue
                print(f'index: {index}, Cropped text: {cropped_text}')
                
                # Display the cropped image (optional for debugging)
                cv2.imshow(f'Index {index}', cropped_image)
                cv2.imshow(f'binary_image', binary_image)

                cv2.waitKey(0)

                detected_ids.append(cropped_text)                   
                   
        
        
        if len(detected_ids) > 0:
            # Find the ID with the maximum number of numeric characters
            largest_id = max(detected_ids, key=lambda x: sum(c.isdigit() for c in x))
            
            # Create a string containing only the numeric characters from the largest ID
            numeric_id = ''.join(c for c in largest_id if c.isdigit())
            
            return numeric_id
            
        # Return the detected ID with the longest number sequence
        return detected_id
    
    def detect_case_number_on_large_sticker(self, words_info, image ):
        
        detected_case_number = ''  # To store the ID with the longest sequence of numbers
        image_height, image_width = image.shape[:2]  # Get image dimensions
        
        detected_ids = []
        for index, word in enumerate(words_info):
            if 'מקר' in word.text and any(c.isdigit() for c in word.text):
                # Check only the next word, if exists
                            
                # Extract the bounding box coordinates of the next word
                x, y, w, h = word.rect

                # Ensure the bounding box stays within the image bounds
                x = max(0, x)  # Clamp x to be at least 0
                y = max(0, y)  # Clamp y to be at least 0
                w = min(w, image_width - x)  # Ensure the width does not go beyond the image
                h = min(h, image_height - y)  # Ensure the height does not go beyond the image


                # Crop the image using the bounding box of the next word
                cropped_image = image[y:y + h, x:x + w]
                # Convert to grayscale
                gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

                # Apply binary thresholding
                _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
                
                # Use pytesseract to extract text from the cropped image
                cropped_text = pytesseract.image_to_string(binary_image, config='--psm 6')
                # Skipping: Contains letters
                # if any(c.isalpha() for c in cropped_text):
                #     continue
                # print(f'index: {index}, Cropped text: {cropped_text}')
                detected_case_number =  ''.join(c for c in cropped_text if c.isdigit())
                
                # Display the cropped image (optional for debugging)
                cv2.imshow(f'Index {index}', cropped_image)
                cv2.imshow(f'binary_image', binary_image)

                cv2.waitKey(0)

                return detected_case_number

            
        return detected_case_number


if __name__ == "__main__":   
   
    # test =  cv2.imread("/home/yakir/sticker_ws/new/test.jpeg",1)   
    # gray_image = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    # _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)    
    # cropped_text = pytesseract.image_to_string(binary_image, config='--psm 6')
    # print(f' cropped_text {cropped_text}')            
    # exit(1)            
    
    ocr_processor = OCRProcessor()   

    image = cv2.imread("/home/yakir/sticker_ws/new/3.jpeg",1)
  
    
    yellow_bgr_color =   (0, 187, 200)  
    cropped_image = ocr_processor.crop_by_color(image, yellow_bgr_color)
    words_info = ocr_processor.process(cropped_image)
    # for word in words_info:
    #     print(word)
    
    full_name = ocr_processor.detect_name(words_info)
    print(f'*** name is: {full_name}')
    id = ocr_processor.detect_person_id(words_info, cropped_image)
    print(f'*** id is: {id}')   
    case_number = ocr_processor.detect_case_number(words_info, cropped_image)
    print(f'*** case_number is: {case_number}')