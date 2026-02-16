Machine Learning System for Eye Tracking and Attention Monitoring

0. Project overview

This project implements an eye-detection system using YOLO to monitor whether the user remains focused on the screen. The model detects eye regions in real time and can be used as a base for attention tracking or fatigue monitoring systems.

As a secondary (and less scientific) feature, the system may trigger a short rickroll when prolonged loss of attention is detected. This serves as a lightweight notification mechanism and a reminder to return focus to the task.

1. Dataset Collection

The dataset was collected in two different lighting environments in order to improve model robustness:

Low-light environment — evening conditions with a single lamp as the primary light source.

Bright environment — a well-lit library with strong and uniform illumination.

Using multiple lighting conditions helps the model generalize better and improves detection accuracy in real-world situations.

Dataset collection was performed using the following scripts:

Save_frames.py — captures and saves frames directly from a webcam.

ConvertVideosToPhotos.py — extracts frames from recorded videos to increase dataset size efficiently.

2. Environment Setup

The following libraries are required before training:

PyTorch with CUDA support

Used for GPU-accelerated training on NVIDIA GPUs.

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

(The CUDA version should match the installed CUDA toolkit and GPU support.)

OpenCV

Used for image processing and dataset preparation.

pip install opencv-contrib-python
Ultralytics (YOLO implementation)
python -m pip install -U ultralytics



3. Dataset Annotation

The dataset must contain label files in YOLO format. Each image has a corresponding .txt file containing normalized bounding box coordinates.

Example:

0 0.503125 0.3572916666666667 0.0875 0.03958333333333333

0 0.653125 0.35833333333333334 0.09375 0.04583333333333333

Each row represents:

<class_id> <x_center> <y_center> <width> <height>

All values are normalized relative to image size.

Bounding boxes were generated automatically using MediaPipe, which provides approximate eye locations. This significantly reduces manual labeling effort.

The annotation process is implemented in:

Create_Boundary_Boxes.py



4. Dataset Structure

The dataset directory follows the standard YOLO layout:

dataset/
    images/
      train/
    labels/
       train/
    data.yaml

Example data.yaml configuration:

path: C:/Users/giley/PycharmProjects/YOLO training/dataset

train: images/train
val: images/train

names:
  0: face


5. Model Training

Training is started using the Ultralytics YOLO command:

yolo detect train model=yolov8n.pt data=dataset/data.yaml device=cuda

Training time depends on hardware and dataset size.
On an NVIDIA RTX 5060 GPU, training with approximately 6900 images took about two hours.

6. Usage

Model testing and real-time inference are implemented in Model_test.py.

The script runs webcam-based detection and displays bounding boxes around detected eye regions in real time.

To run the test:

python Model_test.py

The application requires access to a webcam.

In Model_test.py I am using rickroll library to get atttention back when I don't look on the screen for a while:

pip install rick-roll

Photo:

![Usage with confidence level](/Photos/Usage.png)

And I also attach video of usage

7. Hardware/Performance

- GPU: NVIDIA RTX 5060 (laptop)
- Training time: ~2 hours
- Dataset size: 6900 images

8. Limitations

- The model was trained only on mine photos and can be very inaccurate with other people
- Under extreme lightning changes (it was trained only in 2 conditionals) may also be inacurate
-Requirs quit good CPU to run in real time

9. Future works
Since this mini project is just learning, there are a lot of different ways to improve project, for instance:

gaze direction

attention scoring

lightweight model for Pi 

Summary

The system uses a YOLO-based object detection model trained on a custom dataset collected under different lighting conditions. MediaPipe was used to automate annotation, and training was performed using Ultralytics YOLO with GPU acceleration to achieve efficient training performance.
