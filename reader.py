import os
from PIL import Image
import pytesseract
import cv2

# Function to extract text (number) from an image
def extract_number_from_image(image_path):
    try:
        # Open the image
        image = Image.open(image_path)
        # Optionally, apply image preprocessing here
        # Convert image to text
        
        text = pytesseract.image_to_string(image)
        # Extract numbers or apply logic to find the number in the middle
        # For simplicity, we assume the OCR extracts numbers directly
        return text
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Function to process all images in a folder
def process_images_in_folder(folder_path):
    # List all image files in the folder (assuming JPEG; adjust as necessary)
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.PNG')]
    numbers = []
    for x in range(1000):
        numbers.append(0)
    for image_file in image_files:
        full_path = os.path.join(folder_path, image_file)
        number = extract_number_from_image(full_path)
        print(number)
        if number is not None:
            print(f"Extracted from {image_file}: {number}")
            print("putting in list: " + str(eval(image_file.split('.')[0].split('_')[1])))
            numbers[eval(image_file.split('.')[0].split('_')[1])] =  number
    return numbers

def crop_images_in_folder(source_folder, destination_folder, crop_rectangle):
    """
    Crops all images in the specified source folder and saves the cropped images to the destination folder.

    Parameters:
    - source_folder: Path to the folder containing the input images.
    - destination_folder: Path where the cropped images will be saved.
    - crop_rectangle: A tuple (left, upper, right, lower) specifying the cropping rectangle.
    """
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # List all image files in the source folder (assuming PNG; adjust as necessary)
    image_files = [f for f in os.listdir(source_folder) if f.endswith('.PNG')]

    for image_file in image_files:
        
        full_input_path = os.path.join(source_folder, image_file)
        full_output_path = os.path.join(destination_folder, image_file)
        
        try:
            with Image.open(full_input_path) as img:
                cropped_img = img.crop(crop_rectangle)
                cropped_img = cropped_img.convert('L')
                inverted_img = Image.eval(cropped_img, lambda x: 255 - x)
                new_size = (int(cropped_img.width * 0.3), int(cropped_img.height * 0.3))
                print(new_size)
                enlarged_img = inverted_img.resize(new_size, Image.ANTIALIAS)
                enlarged_img.save(full_output_path)


                print(f"Cropped image saved to {full_output_path}")
        except Exception as e:
            print(f"Error processing {full_input_path}: {e}")


def extract_frames(video_path, output_folder, frames_per_second=4):
    """
    Extracts frames from a video file at the specified rate and saves them to an output folder.

    Parameters:
    - video_path: Path to the video file.
    - output_folder: Folder where extracted frames will be saved.
    - frames_per_second: Number of frames to extract per second of video.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = video.get(cv2.CAP_PROP_FPS)  # Get frames per second of the video
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print(f"Video FPS: {fps}")
    print(f"Total frames: {frame_count}")
    print(f"Duration (s): {duration}")

    # Calculate the interval between the frames you want to capture
    interval = int(fps / frames_per_second)

    frame_id = 0
    num = 0
    while True:
        success, frame = video.read()
        if not success:
            break  # Reached the end of the video

        # Save frame if it matches the desired interval
        if frame_id % interval == 0:
            frame_file = os.path.join(output_folder, f"frame_{num}.PNG")
            cv2.imwrite(frame_file, frame)
            print(f"Saved {frame_file}")
            num += 1

        frame_id += 1

    video.release()
    print("Finished extracting frames.")

def cleanNum(numbers):
    list1 = []
    for x in numbers:
        y = 0
        if (type(x) != int and x != ''):
            print(x)
            print(type(x))
            x = x.replace('\n', '')
            x = x.replace(' ', '')
            print(x)
            y = eval(x)
            y = x
            list1.append(y)
        else:
            list1.append(x)

    for x in range(len(numbers)-4):
        print("x: " + str(x))
        num = 0
        temp = 0
        if (list1[x]):
            temp += eval(list1[x])
            num += 1
        if (list1[x+1]):
            temp += eval(list1[x+1])
            num += 1
        if (list1[x+2]):
            temp += eval(list1[x+2])
            num += 1
        if (list1[x+3]):
            temp += eval(list1[x+3])
            num += 1
        if (list1[x+4]):
            temp += eval(list1[x+4])
            num += 1
        if (num != 0):
            list1[x] = temp/num
        else:
            list1[x] = 0
    return list1

import pandas as pd

def create_excel_from_string(numbers, output_file_name):
    """
    Takes a string of comma-separated numbers and creates an Excel spreadsheet.

    Parameters:
    - number_string: A string containing comma-separated numbers.
    - output_file_name: The name of the Excel file to be created (including the .xlsx extension).
    """

    # Create a Pandas DataFrame
    df = pd.DataFrame(numbers, columns=['Numbers'])

    # Write the DataFrame to an Excel file
    df.to_excel(output_file_name, index=False)

    print(f"Excel file '{output_file_name}' has been created.")

# Example usage
video_path = './HRVids/HRVid1.mp4'
output_folder = './images'

extract_frames(video_path, output_folder)

# Example usage
source_folder = './images'
destination_folder = './clean-images'
# crop_rectangle = (330, 1100, 850, 1300)  # Example rectangle; adjust to your needs
crop_rectangle1 = (240, 580, 650, 770)  # Example rectangle; adjust to your needs
crop_rectangle2 = (100, 580, 700, 775)  # Example rectangle; adjust to your needs

crop_images_in_folder(source_folder, destination_folder, crop_rectangle1)

# Example usage
folder_path = './clean-images'
extracted_numbers1 = process_images_in_folder(folder_path)
extracted_numbers1 = cleanNum(extracted_numbers1)
print("Extracted numbers:", extracted_numbers1)

for x in range(len(extracted_numbers1)):
    print(str(x) + ": " + str(extracted_numbers1[x]))


create_excel_from_string(extracted_numbers1, 'extracted_numbers.xlsx')