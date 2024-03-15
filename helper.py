from PIL import Image
import numpy as np
import easyocr
import google.generativeai as genai
import streamlit as st
from gtts import gTTS

def to_grayscale(image , fast = True) :
    '''
    Converts the input image to grayscale using the formula:

    Y = 0.2989 * R + 0.5870 * G + 0.1140 * B

    where R, G, and B are the red, green, and blue channels of the image, respectively.

    Args:
        1) image : PIL.Image : The input image

    Returns: 
        1) PIL.Image : The grayscale image
    '''
    if fast : image = image.convert('L')
    else :  

        image_data = image.load()

        for row in range(image.width) : 

            for col in range(image.height) : 

                red , green , blue = image_data[row , col]

                red_gray = 0.2989 * red
                green_gray = 0.5870 * green
                blue_gray = 0.1140 * blue

                gray = int(red_gray + green_gray + blue_gray)

                image_data[row , col] = (gray , gray , gray)

    return image

def binary_threshold(image , threshold = 128 , fast = True) : 
    '''
    Converts the input image to binary using the given threshold.

    Formula :
        if pixel < threshold : pixel = 0
        else : pixel = 255

    Args:
        1) image : PIL.Image : The input image
        2) threshold : int : The threshold value

    Returns:
        1) PIL.Image : The binary image
    '''

    if fast : image = image.point(lambda pixel : 0 if pixel < threshold else 255 , '1')
    else :  

        image_data = image.load()

        for row in range(image.width) : 

            for col in range(image.height) : 

                gray_value = image_data[row , col][0]

                if gray_value < threshold : image_data[row , col] = (0 , 0 , 0)
                else : image_data[row , col] = (255 , 255 , 255)

    return image

def bilinear_filter(image , scale_factor , fast = True) :
    '''
    Resizes the input image using bilinear interpolation.

    Formula :
        For each pixel in the new image, we find the corresponding pixel in the original image and use the average of the 4 pixels around it.

    Args:
        1) image : PIL.Image : The input image
        2) scale_factor : float : The scale factor
    
    Returns:
        1) PIL.Image : The resized image
    '''

    if fast : image = image.thumbnail((300 , 300) , Image.BILINEAR)
    else : 

        original_width, original_height = image.size

        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        resized_img = Image.new("RGB", (new_width, new_height))

        original_data = image.load()
        resized_data = resized_img.load()

        for row in range(new_width) : 
            for col in range(new_height) : 

                x = row / scale_factor
                y = col / scale_factor

                x1, y1 = int(x), int(y)
                x2, y2 = min(x1 + 1, original_width - 1), min(y1 + 1, original_height - 1)

                top_left = original_data[x1, y1]
                top_right = original_data[x2, y1]
                bottom_left = original_data[x1, y2]
                bottom_right = original_data[x2, y2]

                interpolated_pixel = (
                    int((1 - (x - x1)) * (1 - (y - y1)) * top_left[0] + (x - x1) * (1 - (y - y1)) * top_right[0] +
                        (1 - (x - x1)) * (y - y1) * bottom_left[0] + (x - x1) * (y - y1) * bottom_right[0]),
                    int((1 - (x - x1)) * (1 - (y - y1)) * top_left[1] + (x - x1) * (1 - (y - y1)) * top_right[1] +
                        (1 - (x - x1)) * (y - y1) * bottom_left[1] + (x - x1) * (y - y1) * bottom_right[1]),
                    int((1 - (x - x1)) * (1 - (y - y1)) * top_left[2] + (x - x1) * (1 - (y - y1)) * top_right[2] +
                        (1 - (x - x1)) * (y - y1) * bottom_left[2] + (x - x1) * (y - y1) * bottom_right[2])
                )

                resized_data[row, col] = interpolated_pixel

    return image


def bicubic_filter(image , scale_factor , fast = True) : 
    '''
    Resizes the input image using bicubic interpolation.

    Formula :
        For each pixel in the new image, we find the corresponding pixel in the original image and use the average of the 16 pixels around it.

    Args:
        1) image : PIL.Image : The input image
        2) scale_factor : float : The scale factor

    Returns:
        1) PIL.Image : The resized image
    '''

    if fast : image = image.thumbnail((300 , 300) , Image.BICUBIC)
    else : 

        original_width, original_height = image.size

        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        resized_img = Image.new("RGB", (new_width, new_height))

        original_data = image.load()
        resized_data = resized_img.load()

        def cubic(pixel):

            cubic_factor = -0.5
            
            if abs(pixel) <= 1 : return (cubic_factor + 2) * abs(pixel) ** 3 - (cubic_factor + 3) * abs(pixel) ** 2 + 1
            elif 1 < abs(pixel) < 2 : return cubic_factor * abs(pixel) ** 3 - 5 * cubic_factor * abs(pixel) ** 2 + 8 * cubic_factor * abs(pixel) - 4 * cubic_factor
            else : return 0

        for row in range(new_width) : 
            
            for col in range(new_height) : 
                
                scaled_row = row / scale_factor
                scaled_col = col / scale_factor

                scaled_row_iter , scaled_col_iter = int(scaled_row) - 1, int(scaled_col) - 1
                x_vals = [scaled_row_iter  + scale for scale in range(4)]
                y_vals = [scaled_col_iter + scale for scale in range(4)]

                contributions = np.zeros((4 , 4 , 3) , dtype = float)

                for x_index , x_val in enumerate(x_vals) : 

                    for y_index , y_val in enumerate(y_vals) : 

                        if 0 <= x_val < original_width and 0 <= y_val < original_height : contributions[x_index , y_index] = np.array(original_data[x_val , y_val])
                        else : contributions[x_index , y_index] = np.array([0 , 0 , 0])

                interpolated_pixel = np.zeros(3 , dtype=float)

                for x_index in range(4) : 

                    for y_index in range(4) : interpolated_pixel += contributions[x_index , y_index] * cubic(scaled_row - scaled_row_iter  - x_index) * cubic(scaled_col - scaled_col_iter - y_index)

                interpolated_pixel = np.clip(interpolated_pixel , 0 , 255)

                resized_data[row , col] = tuple(map(int , interpolated_pixel))

    return resized_img

def get_text_from_image(image , fast = True) : 
    '''
    Extracts the text from the input image using OCR and then generates a response using the Gemini.

    Args:

        1) image : PIL.Image : The input image

    Returns:

        1) str : The response generated by the Gemini
    '''

    if fast : 

        image = Image.open(image)
        image.save('Image.jpg')

        reader = easyocr.Reader(['en'])
        text = reader.readtext('Image.jpg')

        st.write('Image Saved')
        
        text = [
            val[1]
            for val 
            in text
        ]

        text = ' '.join(text)

        st.write('Extracted Images , Please wait for the analysis')

        genai.configure(api_key = 'AIzaSyDyaY8u-BdM_IxU_YbJ5n4JLHy6A4uvxOE')
        model = genai.GenerativeModel('gemini-pro')

        prompt = f'''
    Consider yourself a doctor. This is a prescription/test report of a Patient
    {text}

    Analyse the Report and provide suggestions and your analysis as text. If there is a difficult word. Try to explain the word in simple language. Add a disclaimer to suggest the pateint to visit a doctor if necessary. Do not use lines like I am not a doctor and other. Use Patient name wherevar necessary. If telling for a desiase, also tell expected symptons that the person might be feeling 
        '''

        response = model.generate_content(prompt)

        try : return response.text
        except : 
            try : response.parts[0]
            except : return 'Sorry I couldn Understand the Prescription'

    else :

        image = to_grayscale(image)
        image = binary_threshold(image)
        image = bilinear_filter(image , 2)
        image = bicubic_filter(image , 2)

        reader = easyocr.Reader(['en'])

        text = reader.readtext(image)

        text = [
            val[1]
            for val 
            in text
        ]

        text = ' '.join(text)

        genai.configure(api_key = st.secrets['API_KEY'])
        model = genai.GenerativeModel('gemini-pro')

        prompt = f'''
    Consider yourself a doctor. This is a prescription/test report of a Patient
    {text}  

    Analyse the Report and provide suggestions and your analysis as text. If there is a difficult word. Try to explain the word in simple language. Add a disclaimer to suggest the pateint to visit a doctor if necessary. Do not use lines like I am not a doctor and other. Use Patient name wherevar necessary. If telling for a desiase, also tell expected symptons that the person might be feeling 
        '''

        response = model.generate_content(prompt)

        try : return response.text
        except : 
            try : response.parts[0]
            except : return 'Sorry I couldn Understand the Prescription'

def get_speech_from_text(text) : 
    '''
    Generates a speech from the input text.

    Args:
        1) text : str : The input text

    Returns:
        1) None
    
    '''

    myobj = gTTS(
        text = text , 
        lang = 'en' , 
        slow = False
    ) 
    
    myobj.save('audio.wav')
