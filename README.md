# Virtual-Try-on-using-Keypoint-Detection

## Introduction
Online fashion shopping is convenient but lacks the ability for users to try on clothes, leading to uncertainty and high return rates. StyleSync addresses this challenge by simulating a real-world fitting experience through AI. Our system enables users to virtually try on clothes using their webcam images, delivering personalized and realistic visuals.


## Objectives
The core objectives of StyleSync are:

To create a virtual try-on platform using static images.

To deliver a seamless try-on experience that visually overlays garments onto the user's image.

To integrate pose estimation, segmentation, and gesture recognition modules.

To improve personalization, interactivity, and realism in online fashion browsing.


## Architecture Modules

User Input Module (webcam image)

Pose Estimation (OpenPose)

Gesture Recognition (MediaPipe with MLP/LSTM)

Image Segmentation

Clothing Item Processing

Clothing Warping (TPS transformation)

Try-On Generation

Output Module for display and gesture-based control


