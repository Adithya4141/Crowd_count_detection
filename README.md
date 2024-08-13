Crowd Count Detection Project Overview
Crowd count detection is a process used to identify and count the number of people within a video frame. This task is often employed in areas such as surveillance, public event management, and safety monitoring, where understanding crowd density and movement is critical. In this project, the detected counts are simultaneously updated in a database, allowing for real-time tracking and analysis.

The project is divided into two primary steps:

1. Training a YOLOv8 Model on a Crowd Detection Dataset
Objective: The first step involves training a YOLOv8 model, a state-of-the-art deep learning algorithm designed for real-time object detection. The model is trained to detect crowds within video frames.
Dataset: The training dataset is sourced from Roboflow, a platform providing various datasets tailored for computer vision tasks. The dataset used here is specifically designed for crowd detection, containing labeled images where people are marked.
Process:
The training process is implemented within the Jupyter notebook named crowd-detection.ipynb.
The notebook provides a detailed workflow for setting up the training environment, downloading the dataset, configuring the YOLOv8 model, and running the training process.
YOLOv8 is an advanced version of the YOLO (You Only Look Once) family of models, known for its balance between speed and accuracy, making it well-suited for real-time applications like crowd detection.
2. Performing Inference on Live Video Feeds Using Trained Weights
Objective: After successfully training the YOLOv8 model, the next step is to apply the trained model weights to detect crowds in real-time video feeds. This process is known as inference.
Implementation:
The inference process is carried out using a Python script named inference.py.
The script utilizes the trained YOLOv8 model to process video frames captured from a live feed (or pre-recorded video).
As the model detects people within each frame, the results are annotated onto the video and simultaneously recorded in a SQLite database. This database is designed to store information such as the count of detected individuals, their classes, and the timestamp of detection.
Outcome: The output of this process includes an annotated video that visually highlights detected individuals, as well as a database that can be queried for further analysis, such as tracking the number of people over time.
