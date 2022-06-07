# Vehicle-Tracking-and-Speed-Estimation

## Overview

The vehicle speed detection system is developed using video processing techniques. In this, a video is captured through a fixed camera with various parameters noted. The video from the camera is processed frame by frame, in which the 1) Each frame is processed with Image Enhancement to improve the characteristics of the image 2) Vehicle is identified with a bounding box applied to the vehicle, to note the movement in each frame 3) Movement of the vehicle is noted with change in pixel 4) Calculating the speed using distance formula considering PPM.

## Architecture

<img width="651" alt="Screen Shot 2022-06-07 at 11 04 29 AM" src="https://user-images.githubusercontent.com/34905922/172414619-57e4615f-c636-4c0f-87dc-ddb06572ef96.png">

## Tools and Technologies

Haar Cascade
cmake==3.120
dlib==19.16.0
numpy==1.15.3
opencv-python==3.4.3.1
python
pytesseract
imutils
pandas
PIL

## Demo of the working system

https://user-images.githubusercontent.com/34905922/172417400-935ce347-b81c-488a-a8df-1285c36b469a.mp4

