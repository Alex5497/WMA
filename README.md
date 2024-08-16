# Video Matching Application (WMA)

This project is a video frame matching application that identifies the best matching images from a folder of reference images using ORB (Oriented FAST and Rotated BRIEF) feature detection. The project uses OpenCV for computer vision tasks such as feature detection, matching, and homography computation.

## Features
- **Feature Detection**: Uses ORB to detect keypoints and descriptors from images.
- **Feature Matching**: Uses the Brute-Force Matcher (BFMatcher) to match descriptors between video frames and reference images.
- **Homography and Alignment**: If sufficient matches are found, the best-matching image is aligned to the video frame using homography transformation.
  
## Requirements
- Python 3.x
- OpenCV
- NumPy

Install the dependencies using:
```bash
pip install opencv-python numpy
```

## Usage
1. Place reference images in the `pliki/` directory.
2. Place the video file in the `dane/` directory.
3. Run the script:
```bash
python main5.py
```
4. The application will display the video frame with the best matching image aligned and outlined.

Press `q` to quit the application.

## Folder Structure
- **dane/**: Contains the video file (e.g., `mov.MOV`).
- **pliki/**: Contains reference images for matching.

## Key Parameters
- **MIN_MATCH_COUNT**: Minimum number of good matches required to compute homography (default is 10).
- **ORB Feature Detection**: Configured using OpenCV's ORB feature detector.
- **BFMatcher**: Used for brute-force descriptor matching with L2 norm and cross-check enabled.

## Output
The application shows the video frame with matched keypoints highlighted. When enough matches are found, a bounding box is drawn around the detected object in the frame.

## Limitations
- The accuracy of matching is dependent on the quality of the reference images and the video.
- The current implementation uses a simple brute-force matching strategy, which might be slow for larger datasets.

## Future Improvements
- Implement a more efficient matching algorithm, such as FLANN-based matcher.
- Add support for real-time camera input.
- Optimize the pipeline for faster processing.

## License
This project is licensed under the MIT License.
