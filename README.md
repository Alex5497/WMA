Image Matching and Perspective Transformation

This project is aimed at matching images from a given video stream to a set of reference images and performing perspective transformation based on the matches found.

Features

Matches video frames with a set of reference images using ORB (Oriented FAST and Rotated BRIEF) feature detection and matching.
Computes homography to find the perspective transformation between the matched points.
Draws the matched features between the reference image and the video frame.
Allows users to specify a minimum number of matches required for perspective transformation.
Requirements

Python 3.x
OpenCV (cv2)
NumPy
Installation

Clone this repository:
bash
Copy code
git clone https://github.com/your_username/image-matching.git
Install the required dependencies:
Copy code
pip install opencv-python numpy
Usage

Place your video file (mov.MOV) in the dane/ directory.
Place the reference images in the pliki/ directory.
Run the Python script:
Copy code
python image_matching.py
Press 'q' to exit the application.
Configuration

Adjust the MIN_MATCH_COUNT variable in the script to set the minimum number of matches required for perspective transformation.
Modify the code to change feature detection and matching algorithms or adjust matching parameters according to your needs.
License

This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments

This project is inspired by computer vision and image processing techniques.
Thanks to the OpenCV and NumPy communities for providing powerful libraries for image processing in Python.
