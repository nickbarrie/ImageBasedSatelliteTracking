import os
import numpy as np
import cv2
from astropy.io import fits
import argparse
import matplotlib.pyplot as plt

# Function to load a FITS file
def load_fits_image(fits_file):
    hdul = fits.open(fits_file)
    return hdul[0].data  # Return the image data

# Function to preprocess the image
def preprocess_image(image_data):
    # Ensure the image is 8-bit (CV_8UC1)
    image_data = np.uint8(image_data)  # Convert to 8-bit if it's not already
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image_data, (5, 5), 0)
    return blurred

# Function to calculate the displacement (motion between frames)
def calculate_motion(prev_image, curr_image):
    # Compute the absolute difference between the two frames
    diff = cv2.absdiff(prev_image, curr_image)

    # Apply threshold to the difference to get a binary image highlighting the object
    _, thresh_diff = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to remove noise and improve the contours
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh_diff, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Find contours to locate the moving object
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If any contours are found, calculate the centroid of the moving object
    if contours:
        for contour in contours:
            # Calculate the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate the centroid of the moving object
            centroid = (x + w // 2, y + h // 2)
            return centroid, (x, y, w, h)  # Return both centroid and bounding box
    return None, None  # No movement detected

# Function to get the two most recent FITS files in the specified directory
def get_most_recent_fits_files(directory):
    # Get a list of all FITS files in the directory
    fits_files = [f for f in os.listdir(directory) if f.endswith('.fits')]
    
    # Get the full path for each file and sort by modification time (most recent first)
    fits_files = sorted([os.path.join(directory, f) for f in fits_files], key=os.path.getmtime, reverse=True)

    if len(fits_files) < 2:
        raise ValueError("Not enough FITS files in the directory.")

    return fits_files[0], fits_files[1]  # Return the two most recent files

# Main function to detect the moving object between two frames
def main(directory):
    # Get the two most recent FITS files in the directory
    prev_fits_file, curr_fits_file = get_most_recent_fits_files(directory)

    # Load the FITS images
    prev_image_data = load_fits_image(prev_fits_file)
    curr_image_data = load_fits_image(curr_fits_file)

    # Preprocess the images
    prev_processed = preprocess_image(prev_image_data)
    curr_processed = preprocess_image(curr_image_data)

    # Calculate the movement of the object
    movement, bounding_box = calculate_motion(prev_processed, curr_processed)

    # Optionally, visualize the frames and detected movement
    plt.subplot(1, 3, 1)
    plt.imshow(prev_processed, cmap='gray')
    plt.title("Previous Frame")

    plt.subplot(1, 3, 2)
    plt.imshow(curr_processed, cmap='gray')
    plt.title("Current Frame")

    plt.subplot(1, 3, 3)
    diff = cv2.absdiff(prev_processed, curr_processed)
    plt.imshow(diff, cmap='hot')
    plt.title("Difference")

    # If movement was detected, draw a red circle around it in the current frame
    if movement:
        # Convert the grayscale image to a 3-channel color image
        circle_image = cv2.cvtColor(curr_processed, cv2.COLOR_GRAY2BGR)  # Convert to BGR (3 channels)

        # Set the parameters for the circle
        radius = 30  # Set the radius of the circle (adjust as needed)
        color = (255, 0, 0)  # Red color in BGR format
        thickness = 6  # Circle thickness

        # Draw the red circle at the detected movement location
        cv2.circle(circle_image, movement, radius, color, thickness)

        # Calculate the center of the current frame
        frame_center = (curr_processed.shape[1] // 2, curr_processed.shape[0] // 2)

        # Calculate the offset between the center of the frame and the centroid of the object
        offset_x = movement[0] - frame_center[0]  # Horizontal offset
        offset_y = movement[1] - frame_center[1]  # Vertical offset

        # Determine the movement direction for telescope adjustment
        if abs(offset_x) > 10:  # Arbitrary threshold for noticeable movement
            if offset_x < 0:
                print("Move telescope left (centroid too far right)")
            else:
                print("Move telescope right (centroid too far left)")

        if abs(offset_y) > 10:  # Arbitrary threshold for noticeable movement
            if offset_y < 0:
                print("Move telescope up (centroid too low)")
            else:
                print("Move telescope down (centroid too high)")

        # Show the frame with the red circle and movement indication
        plt.subplot(1, 3, 3)
        plt.imshow(circle_image)
        plt.title("Movement with Red Circle")

    plt.show()

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Detect movement between two recent FITS files.")
    parser.add_argument("directory", type=str, help="Directory containing the FITS files")
    args = parser.parse_args()

    # Run the main function with the provided directory
    main(args.directory)
