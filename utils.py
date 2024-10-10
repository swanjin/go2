import sys
import signal
import textwrap
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

def alarm_handler(signum, frame):
    raise TimeoutError

def timeoutInput(timeout):
    s = ""
    if timeout is not None:
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(timeout)
    try:
        while True:
            c = sys.stdin.read(1)
            s += c
            if c == '\n':
                break
    except TimeoutError as err:
        return s
    finally:
        if timeout is not None:
            signal.alarm(0)
    return s

def put_text_in_the_middle(image, text, width, height, font_size=40):
    """
    Function to create an image and place the given text in the middle.

    Parameters:
    text (str): The text to display.
    width (int): The width of the image.
    height (int): The height of the image.
    font_size (int): The size of the font for the text.

    Returns:
    numpy array: The image with text centered, ready for display with imshow.
    """
    # Create a new image with black background
    draw = ImageDraw.Draw(image)

    # Try to load a larger font, fall back to default if necessary
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Get the size of the text for positioning
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Calculate the position to center the text
    position = ((width - text_width) // 2, (height - text_height) // 2)

    # Add the text to the image in white
    draw.text(position, text, fill="white", font=font)

    return image

def combine_images_vertically(self, image1, image2):
    # Get the dimensions of the images
    width1, height1 = image1.size
    width2, height2 = image2.size
    
    # Determine the width and the total height of the combined image
    combined_width = max(width1, width2)
    combined_height = height1 + height2
    
    # Create a new blank image with the appropriate size
    combined_image = Image.new("RGB", (combined_width, combined_height))
    
    # Paste the first image at the top
    combined_image.paste(image1, (0, 0))
    
    # Paste the second image below the first
    combined_image.paste(image2, (0, height1))
    
    return combined_image

def image_text_append(image, image_width, image_height, text):
    text_width = image_width//2
    text_height = image_height

    combined_img = Image.new('RGB', (image_width+text_width, image_height))
    combined_img.paste(image, (0, 0))

    # Create an image for text
    text_img = Image.new('RGB', (text_width, text_height), 'white')
    draw = ImageDraw.Draw(text_img)
    
    # Define the font and size
    try:
        # Adjust path to font as necessary
        font = ImageFont.truetype('arial.ttf', 15)
    except IOError:
        font = ImageFont.load_default()

    # Wrap text
    wrapped_text = [textwrap.fill(line, width=80).split('\n') for line in text.split('\n')]

    # Calculate initial text position
    text_y = 10  # Start 10 pixels from the top
    text_x = 10  # Start 10 pixels from the left for left alignment

    for line in wrapped_text:
        draw.text((text_x, text_y), line, fill='black', font=font)
        # Get height of the current line to adjust the vertical position for the next line
        text_size = draw.textsize(line, font=font)
        text_y += text_size[1]  # Move down to draw the next line  
    # Debug save the text image alone to verify text appearance
    # text_img.save(os.path.join(self.save_dir, "debug_text_image.jpg"))

    combined_img.paste(text_img, (image_width, 0))
    return combined_img

def PIL2OpenCV(pil_image): 
    numpy_image= np.array(pil_image)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return opencv_image # NumPy array

def OpenCV2PIL(opencv_image):
    color_coverted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_coverted)
    return pil_image
