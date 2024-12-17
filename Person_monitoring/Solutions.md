# MediaPipe 

[MediaPip](https://ai.google.dev/edge/mediapipe/solutions/guide) is a suite of tools developed by Google for applying machine learning techniques in our applications. In the following, we are going to consider more a solution of MediaPip named [Face landmark detection](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker). The MediaPipe Face Landmarker task lets you detect face landmarks and facial expressions in images and videos.  The task outputs 3-dimensional face landmarks, blendshape scores (coefficients representing facial expression) to infer detailed facial surfaces in real-time. [Here is a python API showing how to use this task.](https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb)
The Face Landmarker uses a series of models to predict face landmarks.
1. **Face detection model**: detects the presence of faces with a few key facial landmarks. [Link to the model card.](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf)
2. **Face mesh model**: adds a complete mapping of the face. The model outputs an estimate of 478 3-dimensional face landmarks. [Link to the model card.](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf)
3. **Blendshape prediction model**: receives output from the face mesh model predicts 52 blendshape scores, which are coefficients representing facial different expressions. [Link to the model card.](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf)

## Face detection: 
 The model that works with single images or a continuous stream of images. The task outputs face locations, along with the following facial key points: left eye, right eye, nose tip, mouth, left eye tragion, and right eye tragion. [Here is the link to the document webpage](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector#blazeface_short-range). There are several variants of this model but the version of this model that is used in the pipline of *Face landmark detection* is named : ***MediaPipe BlazeFace Short Range***. A lightweight model for detecting single or multiple faces within selfie-like images from a smartphone camera or webcam. The model is optimized for front-facing phone camera images at short range. This model is based on a paper([SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)) published on *29 Dec 2016*. [Here is the link to github implementation of the paper](https://github.com/weiliu89/caffe/tree/ssd). In the following, there are some key points of the paper: 

 - 	**Single-Shot Approach**: SSD eliminates region proposal steps; directly predicts object categories and bounding box offsets from feature maps.
 -  **Multi-scale Feature Maps**: Predictions are made at multiple scales by applying convolutional filters to feature maps of various resolutions.
 -  **Default Boxes**: Predefined bounding boxes of different scales and aspect ratios are used to match ground-truth boxes during training.
 -  **Unified Framework**: Combines all computations in a single deep neural network for object detection without intermediate resampling.

 - **Improvement over YOLO**: Unlike YOLO, SSD utilizes multi-scale feature maps, enabling better detection of small objects and higher accuracy (74.3% mAP for SSD300 vs. 63.4% mAP for YOLO).
 - **Efficiency with Accuracy**: SSD provides accuracy comparable to region proposal-based methods (e.g., Faster R-CNN) while being significantly faster.
 - SSD300 (300×300 pixels as input images) achieves 74.3% mAP on VOC2007, surpassing YOLO and Faster R-CNN.
 - Larger input models (SSD512) achieve up to 81.6% mAP on VOC2007 and 80% on VOC2012.
 - SSD300 runs at 59 FPS, while SSD512 achieves near real-time speed at 22 FPS.
 - Base Network: SSD is built on a modified VGG16, optimized for object detection tasks.
 - **Default Box Flexibility**: Supports a wide range of scales and aspect ratios, allowing better handling of diverse object shapes and sizes.
 - **Real-Time Application**: SSD300 is the first real-time detection model to exceed 70% mAP.


## Face Mesh Model: 
A lightweight model to predict 3D facial surface landmarks from monocular video captured by a front-facing camera on a smartphone in real time. This model is based on [this paper](https://arxiv.org/abs/2006.10962) published on *19 Jun 2020*. The paper presents Attention Mesh, a new method for predicting a 3D face mesh from a 2D image in real-time. The key innovation is the use of an attention mechanism to focus on important facial regions like the eyes and lips, improving accuracy without sacrificing speed. Here there are some key points about the paper: 

- **Input**: The model takes a 256x256 pixel image of a face as input. This image can come from a face detector or a tracking system.
- **Feature Extraction**: The image is processed by a convolutional neural network to extract a 64x64 feature map.
- **Initial Mesh Prediction**: One submodel uses the feature map to predict the locations of all 478 face mesh vertices in 3D. This provides a rough estimate of the face mesh and is used to define crop bounds for the attention mechanism.
- **Attention Mechanism**: The model uses a spatial transformer network to focus on three key regions: the lips and two eyes.
- **Region-specific Heads**: Separate submodels are used to predict the landmarks for each region. Each eye submodel also predicts the iris as a separate output.
- **Spatial Transformer**: The spatial transformer module extracts 24x24 feature maps for each region from the 64x64 feature map. It does this by applying an affine transformation controlled by a matrix that allows zooming, rotating, translating, and skewing the sampled grid of points.
- **Region Refinement**: The region-specific submodels use the extracted features to refine the initial mesh prediction for each region.
- **Normalization**: Normalization is applied to ensure the eyes and lips are aligned horizontally and have a uniform size, further improving accuracy.
- **Output**: The model outputs a refined 3D face mesh with accurate landmarks, especially in the crucial eye and lip regions.
- **Two-Phase Training**: The Attention Mesh network is trained in two phases:
    - **Phase 1**: The submodels are trained independently using ideal crops from ground truth data with slight augmentations.
    - **Phase 2**: The submodels are trained again using crop locations predicted by the model itself. This helps the region submodels adapt to the model's own predictions, making it more robust.
- **Efficiency Gains**: Reduces the need for CPU-GPU synchronization and inference time by performing the entire process on the GPU.
- **Real-Time Capability**: Designed for real-time on-device use, achieving over 50 FPS on a Pixel 2 device.
- Inference time is 16.6 ms for the Attention Mesh, versus 22.4 ms for cascaded models (approximately 30% faster).
- **Inference Speed**: 25%+ faster than cascaded models due to unified architecture and reduced synchronization overhead.
- **Mesh Quality**: Comparable or slightly superior quality, especially in eye regions.


## BlendShape Prediction Model : 
A lightweight model to predict 52 facial blendshapes from facial landmarks in realtime. Facial landmarks used by this model frst need to be produced by the FaceMesh model that runs on monocular video. The model is based on [the paper](https://arxiv.org/abs/2309.05782) published on *11 Sep 2023*. The paper presents Blendshapes GHUM – an on-device ML pipeline that predicts 52 facial blendshape coefficients at 30+ FPS on modern mobile phones. The main contributions are: i) an annotation-free offline method for obtaining blendshape coefficients from real-world human scans, ii) a lightweight real-time model that predicts blendshape coefficients based on facial landmarks. Here there are some key points about the method: 

- **Training Dataset Preparation**
    - **3D Scanning**:
	    - Captured high-resolution 3D scans of 6,000 individuals performing 40 predefined expressions (e.g., smiling, frowning).
	    - Each scan starts with a neutral expression and transitions to the target expression.
    - **Template Registration**:
	    - Aligned 3D scans to a canonical template mesh (12,201 vertices) to standardize topology and semantics.
	- **Blendshape Transfer**:
	    - Transferred 52 canonical blendshapes to each individual’s neutral mesh using affine deformation transfer to personalize them.
	- **Blendshape Coefficient Optimization**:
	    - Solved for blendshape coefficients to reconstruct each scanned expression using the blendshape model.
	- **Synthetic Data Augmentation**:
	    - Generated additional expressions by sampling realistic combinations of blendshape coefficients.
	    - Projected 3D data into 2D landmarks using known camera parameters.
	- **Final Dataset Size**:
	    - Approximately 2 million pairs of 2D facial landmarks (146 \times 2) and corresponding blendshape coefficients (1 \times 52).

- **Training Steps**
	1.	Train a facial landmark detection model (MobileNetV2-based) to extract 478 2D landmarks from input RGB images.
	2.	Extract 146 key landmarks (lips, eyes, eyebrows, irises, and face contours) to serve as input to the blendshape model.
	3.	Train the blendshape prediction model using:
	    - L2 Loss on Blendshape Coefficients: Ensures numerical accuracy.
	    - Perceptual Loss: Minimizes landmark prediction errors from reconstructed 3D meshes for perceptual realism.
- **Model Architecture**
    - **Facial Landmark Model**
	    -  Based on MobileNetV2.
	    - Predicts 478 landmarks from 256×256 input images.
	    - Achieved a Mean Normalized Error (MNE) of 2.71% on test data.
	- **Blendshape Prediction Model**
	    - Uses a lightweight MLP-Mixer architecture.
	    - Input: 146 \times 2 2D landmarks.
	    - Output: 52 blendshape coefficients and a 6D facial rotation matrix.
	    - Includes latent representation of 96 \times 64 for intermediate computations.
- **Performance and Time Efficiency**
    - **Landmark Detection**:
	    - Runs at 8ms per frame on a Pixel 6 phone using TensorFlow Lite OpenCL backend.
	- **Blendshape Prediction**:
	    - Runs at 1.2ms per frame on a Pixel 6 phone using TensorFlow Lite XNNPACK backend.
	- **Overall Pipeline Speed**:
	    - Achieves real-time performance at 30+ FPS on modern mobile devices.
- **Accuracy**:
    - Real-time blendshape model achieved an MNE of 3.88%, comparable to human-level annotation accuracy (MNE of 2.33%).
	- 96% success rate in capturing expressions during qualitative user studies.
	- Model accuracy significantly outperforms errors between artist-designed blendshapes and canonical templates (MNE ~8.45%).


## MediaPipe Python Framework: 

MediaPipe is a cross-platform framework by Google that simplifies the development of **real-time media processing pipelines**. Whether you're building face detection, object detection, pose tracking, or other computer vision tasks, MediaPipe’s modular “calculator graph” architecture lets you chain together operations to form a complete ML pipeline. Although MediaPipe’s core is written in C++, **the Python bindings** provide an easy-to-use API for rapid prototyping and integration.

This document aims to give you a **thorough understanding** of how MediaPipe Python works, covering **core concepts**, the **high-level Solutions API**, and **custom graph usage**. 

---

### **Table of Contents**
1. [Core Concepts](#core-concepts)
2. [MediaPipe Architecture](#mediapipe-architecture)
3. [Python Solutions API](#python-solutions-api)
    - [Face Detection Example](#face-detection-example)
    - [Face Mesh Example](#face-mesh-example)
4. [How TensorFlow Lite is Utilized](#how-tensorflow-lite-is-utilized)
5. [Custom Graphs & Calculators](#custom-graphs--calculators)
    - [Writing a Custom `.pbtxt` Graph](#writing-a-custom-pbtxt-graph)
    - [Running a Custom Graph in Python](#running-a-custom-graph-in-python)
6. [Post-Processing & Custom Logic](#post-processing--custom-logic)

---

### **Core Concepts**

1. **Calculator**:  
   - A **C++ class** that processes data (e.g., images, tensors, bounding boxes).  
   - Examples: `ImageTransformationCalculator` (resize/normalize), `TfLiteInferenceCalculator` (run TFLite models), `TfLiteTensorsToDetectionsCalculator` (decode model outputs).

2. **Graph**:  
   - A `.pbtxt` configuration describing how calculators connect in a **directed acyclic graph** (DAG).  
   - Each node references a calculator, defines input/output streams, and sets options (model paths, scaling factors, etc.).

3. **Packets**:  
   - Discrete data units flowing through streams in the graph (e.g., frames, detection lists).

4. **Solutions**:  
   - High-level Python APIs that come **pre-packaged** with default `.pbtxt` graphs and TFLite models (face detection, face mesh, hand tracking, pose, etc.).  
   - They hide the complexity of the underlying graph.

---

### **MediaPipe Architecture**

**MediaPipe** orchestrates data flow with **calculators** linked in a graph: 

Input (Camera/Image) –> [Calculator] –> [Calculator] –> … –> Output (Detections / Landmarks)

- The pipeline is typically defined in **`.pbtxt`**.  
- Each **calculator** is compiled in C++ and referenced by name in that config.  
- When using **Python Solutions**, the graph is loaded behind the scenes; you just call `.process()` on your frames.

---

## **Python Solutions API**

The simplest approach for most tasks is to use the **high-level solutions** in Python, such as `mp.solutions.face_detection`, `mp.solutions.face_mesh`, `mp.solutions.hands`, etc. These solutions:

- Initialize a **CalculatorGraph** internally.  
- Load a .tflite model and relevant calculators.  
- Provide a `.process()` method to run inference on images.

### **Face Detection Example**

```python
import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize Face Detection with default model.
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,             # 0 for short-range, 1 for full-range
    min_detection_confidence=0.5
)

image_path = 'face.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = face_detection.process(image_rgb)

if results.detections:
    for detection in results.detections:
        # Each detection has bounding box & keypoints
        box = detection.location_data.relative_bounding_box
        print("Bounding box:", box)
        
        # Draw bounding boxes (optional)
        mp_drawing.draw_detection(image, detection)

cv2.imshow("Face Detection", image)
cv2.waitKey(0) 
```

### **Face Mesh Example**
```python
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

image = cv2.imread('face.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = face_mesh.process(image_rgb)
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        print(f"Found {len(face_landmarks.landmark)} landmarks.")
```

## What’s happening under the hood?
- A C++ graph with multiple calculators (image transformation, TFLite inference, landmark decoding).
- The final output is a list of NormalizedLandmarkList protos for each face.

## How TensorFlow Lite is Utilized

MediaPipe uses TensorFlow Lite internally for lightweight inference on CPU/GPU across platforms. The TfLiteInferenceCalculator:
1.	Loads the .tflite model.
2.	Sets up an interpreter (optionally with GPU or NNAPI delegates).
3.	Runs inference on each frame/tensor.
4.	Passes outputs downstream (e.g., bounding boxes, heatmaps, landmarks).

Example: The FaceDetection solution has one TFLite model for bounding box detection, plus additional post-processing calculators for confidence thresholding and non-max suppression.


## Custom Graphs & Calculators

If you need custom logic—like using your own TFLite model or different post-processing—you can define your own .pbtxt graph and run it from Python, bypassing the high-level solutions.

### Writing a Custom .pbtxt Graph

Example my_custom_graph.pbtxt:

``` json 
# Reads frames from an image, transforms, runs TFLite, outputs detections.
node {
  calculator: "ImageDecoderCalculator"
  input_side_packet: "FILE_PATH:image_path"
  output_stream: "IMAGE:decoded_image"
}

node {
  calculator: "ImageTransformationCalculator"
  input_stream: "IMAGE:decoded_image"
  output_stream: "IMAGE:preprocessed_image"
  options: {
    [mediapipe.ImageTransformationOptions.ext] { scale: 0.5 }
  }
}

node {
  calculator: "TfLiteInferenceCalculator"
  input_stream: "TENSORS:preprocessed_image"
  output_stream: "TENSORS:raw_outputs"
  options: {
    [mediapipe.TfLiteInferenceCalculatorOptions.ext] {
      model_path: "path/to/your_model.tflite"
    }
  }
}

node {
  calculator: "TfLiteTensorsToDetectionsCalculator"
  input_stream: "TENSORS:raw_outputs"
  output_stream: "DETECTIONS:final_detections"
}

output_stream: "DETECTIONS:final_detections"
```

### Running a Custom Graph in Python

```python
import mediapipe as mp
import google.protobuf.text_format
from mediapipe.framework import calculator_pb2

def run_custom_graph(pbtxt_path, image_path):
    # Parse graph config
    graph_config = calculator_pb2.CalculatorGraphConfig()
    with open(pbtxt_path, 'r') as f:
        pbtxt_text = f.read()
    google.protobuf.text_format.Merge(pbtxt_text, graph_config)

    # Create and initialize the graph
    graph = mp.CalculatorGraph(graph_config)

    # Prepare poller for the output stream
    poller = graph.AddOutputStreamPoller("final_detections")

    graph.StartRun({})
    
    # Provide side packet data if needed
    side_packet_data = {"image_path": mp.packet_creator.String(image_path)}
    graph.CloseAllPacketSources()  # If you only have side packets, no more data needed
    graph.WaitUntilIdle()

    # Retrieve the output
    packet = None
    if poller.Next(&packet):
        detections = packet.Get()  # The type depends on the graph output (e.g., Detections proto)
        print("Detections:", detections)
    else:
        print("No detections output received.")

    graph.WaitUntilDone()

# Usage
run_custom_graph("my_custom_graph.pbtxt", "path/to/face.jpg")

```


This code:
1.	Loads your .pbtxt into a CalculatorGraphConfig.
2.	Creates a CalculatorGraph.
3.	Adds a poller for the output stream "final_detections".
4.	Passes side packets (like an image path) to the graph.
5.	Waits for the graph to process and retrieves the output.

## Post-Processing & Custom Logic

If you want to change non-max suppression or customize bounding box decoding, you have two main approaches:
1.	Manual Post-Processing in Python:
- Let MediaPipe produce raw bounding boxes or detection proposals, then apply your own NMS in Python.
2.	Create/Modify a Custom Calculator:
- Clone the MediaPipe repo, write a new C++ calculator or fork the existing NonMaxSuppressionCalculator, and reference it in .pbtxt.
- This method keeps everything within the graph pipeline for potential performance benefits.


---
---

# InsightFace 

This project is mainly focused on Face Detection and Face Recognition tasks while it also includes some features for Facial Landmark Detection and Facial Attribute Classification. The two main publication that this library is based on for Face Detection / Localization are : [**RetinaFace** (CVPR 2020)](https://arxiv.org/abs/1905.00641) and [**SCRFD** (arXiv 2021)](https://arxiv.org/abs/2105.04714). 

