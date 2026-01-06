# AI Research

> For my final paper "Using Artificial Intelligence for Computer Vision in Autonomous Vehicles" 2025

## About

This is my go at road sign detection with neural networks. I coded a neural network from scratch at first (`neural_network.py`) but then quickly switched to tensorflow because training with my own model was very unstable and programming a network (+ optimizers and everything) from ground up is not the focus of my paper anyway. `classification.ipynb` contains code for training an MLP and CNN on the GTSRB dataset. The models I trained reached ~ 90-98% validation accuracy - there's a link for downloading them further down. I also experimented with creating nice visuals for my paper in the Notebook, displaying what a CNN "sees". The second Notebook `yolo_and_segmentation.ipynb` trains YOLO and a Kind-of-Encoder-Decoder-Model for Segmentation on data that I generated and labeled myself. The raw images and dataset are accessible through the Drive download link. The coolest part is applying YOLO to a live webcam feed (`live_yolo_annotation.py`) and overlaying the model's predictions on a video.

## Models and own dataset:

[Google Drive](https://drive.google.com/drive/folders/15utBzsVSlDbJRRMLAO_xTXw8hY7N2fCB?usp=sharing)

## Find my paper (in German) at:

Coming soon

## Dataset used

J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel, Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition, Neural Networks, Available online 20 February 2012, ISSN 0893-6080, 10.1016/j.neunet.2012.02.016. (http://www.sciencedirect.com/science/article/pii/S0893608012000457) Keywords: Traffic sign recognition; Machine learning; Convolutional neural networks; Benchmarking

## Pretrained Models

[Ultralytics YOLOv5](https://pytorch.org/hub/ultralytics_yolov5/) [Download: 22.02.2025]
