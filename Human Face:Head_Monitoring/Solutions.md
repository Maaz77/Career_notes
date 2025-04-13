# Table of Contents

- [MediaPipe](#mediapipe)
  - [Face detection](#face-detection)
  - [Face Mesh Model](#face-mesh-model)
  - [BlendShape Prediction Model](#blendshape-prediction-model)
  - [MediaPipe Python Framework](#mediapipe-python-framework)
    - [Core Concepts](#core-concepts)
    - [MediaPipe Architecture](#mediapipe-architecture)
    - [Python Solutions API](#python-solutions-api)
    - [Custom Graphs & Calculators](#custom-graphs--calculators)
    - [Post-Processing & Custom Logic](#post-processing--custom-logic)
- [InsightFace](#insightface)
  - [RetinaFace](#retinaface)
    - [Method](#method)
    - [Dense Regression Branch](#dense-regression-branch)
    - [Accuracy](#accuracy)
    - [Time Efficiency](#time-efficiency)
    - [Key Highlights](#key-highlights)
  - [SCRFD](#scrfd)
    - [Overview](#1-overview)
    - [Key Methods Proposed](#2-key-methods-proposed)
    - [Efficiency](#3-efficiency)
    - [Accuracy](#4-accuracy)
    - [Time Complexity (Inference Speed)](#5-time-complexity-inference-speed)
    - [Key Contributions](#6-key-contributions)
    - [Practical Significance](#7-practical-significance)
    - [Key Takeaways](#8-key-takeaways)
    - [Performance Summary](#9-performance-summary)
    - [Conclusion](#10-conclusion)
  - [BlazeFace](#blazeface)
- [dlib](#dlib)
- [YOLO5FACE paper (2022)](#yolo5face-paper-2022)
  - [YOLO5Face Model Versions and Naming Convention](#1-yolo5face-model-versions-and-naming-convention)
  - [YOLO5Face Performance Metrics](#2-yolo5face-performance-metrics)
  - [Comparison with RetinaFace and SCRFD](#3-comparison-with-retinaface-and-scrfd)
  - [Summary Table](#4-summary-table)
  - [Conclusions](#5-conclusions)
  - [YOLO5Face: Brief Description](#yolo5face-brief-description)
  - [Summary of the Discussion on YOLOv1 Training and Concepts](#summary-of-the-discussion-on-yolov1-training-and-concepts)
  - [Differences Between YOLO5Face and SSD Framework](#differences-between-yolo5face-and-ssd-framework)
- [EfficientFace Paper (2023)](#efficientface-paper-2023)
  - [Overview of EfficientFace](#1-overview-of-efficientface)
  - [Architecture Details](#2-architecture-details)
  - [Performance](#3-performance)
  - [Efficiency](#4-efficiency)
  - [Time and Memory Complexity](#5-time-and-memory-complexity)
  - [Training and Implementation Details](#6-training-and-implementation-details)
  - [Key Observations](#7-key-observations)
  - [Summary](#8-summary)
- [OpenVino](#openvino)
  - [Key Features](#key-features)
  - [Workflow Overview](#workflow-overview)
  - [Additional Information](#additional-information)
- [ULTRA-LIGHT github repo](#ultra-light-github-repo)
- [ONNX model zoo (Ultra-lightweight face dectector)](#onnx-model-zoo-ultra-lightweight-face-dectector)
- [PINTO MODEL ZOO](#pinto-model-zoo)
  - [DSDF VGG](#dsfd-vgg)
  - [DB Face](#dbface)
  - [Face Landmark](#face-landmark)
  - [Iris Landmark](#iris-landmark)
  - [CenterFace](#centerface)
  - [RetinaFace](#retinaface)
  - [WHENet](#whenet)
  - [SCRFD](#scrfd)
  - [Head Pose Estimation Adas-0001](#head-pose-estimation-adas-0001)
  - [YuNet](#yunet)
  - [Face Detection Adas-0001](#face-detection-adas-0001)
  - [Lightweight Pose Estimation](#lightweight-pose-estimation)
  - [6DRepNet](#6drepnet)
  - [SLPT](#slpt)
  - [FAN](#fan)
  - [SynergyNet](#synergynet)
  - [DMHead](#dmhead)
  - [HHP Net](#hhp-net)
  - [YOLOV7 head](#yolov7-head)
  - [DirectMHP](#directmhp)
  - [RetinaFace MobileNetV2](#retinaface-mobilenetv2)
  - [FaceMeshV2](#facemeshv2)
  - [STAR](#star)
  - [6DRepNet360](#6drepnet360)
  - [FaceBox](#facebox)
  - [MobileFaceNet](#mobilefacenet)
  - [Peppa Pig Face Landmark](#peppa-pig-face-landmark)
  - [PipeNet](#pipnet)
  - [Opal23 Head Pose](#opal23-head-pose)
- [Benchmarking of PINTO Models on STM32 MCU with Neural-ART™](#benchmarking-of-pinto-models-on-stm32-mcu-with-neural-art)
- [ONNX model zoo](#onnx-model-zoo)
- [B-FPGM ( Latest face detector, 2025)](#b-fpgm--latest-face-detector-2025)
- [FeatherFace ( Latest face detector, 2025)](#featherface--latest-face-detector-2025)
- [Comparative Analysis of CenterFace and BlazeFace for Face Detection Applications](#comparative-analysis-of-centerface-and-blazeface-for-face-detection-applications)


<br><br>


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
	    - Approximately 2 million pairs of 2D facial landmarks (146 * 2) and corresponding blendshape coefficients (1 * 52).

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
	    - Input: (146 * 2) 2D landmarks.
	    - Output: 52 blendshape coefficients and a 6D facial rotation matrix.
	    - Includes latent representation of 96 * 64 for intermediate computations.
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

<br><br>

# InsightFace 

This project is mainly focused on Face Detection and Face Recognition tasks while it also includes some features for Facial Landmark Detection and Facial Attribute Classification. The two main publication that this library is based on for Face Detection / Localization are : [**RetinaFace** (CVPR 2020)](https://arxiv.org/abs/1905.00641), [**SCRFD** (arXiv 2021)](https://arxiv.org/abs/2105.04714), and [**BlazeFace** ( arXiv 2019)](https://arxiv.org/abs/1907.05047). [**Here**](https://github.com/deepinsight/insightface/tree/master/model_zoo) is the link to the model zoo of this project.

## RetinaFace: 

### **Method**
- **Objective**: RetinaFace is a **single-stage dense face localization** method that detects faces, aligns facial landmarks, and predicts dense 3D face correspondence.
- **Key Components**:
   - **Multi-task Learning**: Simultaneously predicts:
     - Face score (classification: face/not face).
     - Bounding box (face localization).
     - Five facial landmarks (eye centers, nose tip, and mouth corners).
     - Dense 3D face vertices (self-supervised dense regression).
   - **Feature Pyramid Network (FPN)**: Multi-scale feature maps (P2 to P6) for detecting faces at different scales.
   - **Context Modules**: Expand receptive field and add contextual information for robust detection, particularly for tiny faces.
   - **Deformable Convolutional Networks (DCN)**: Handle non-rigid facial transformations and improve context modeling.
   - **Dense Regression Branch**: Predicts dense 3D face mesh with a **Mesh Decoder** using graph convolutions and a differentiable renderer.



### **Dense Regression Branch**
- **Purpose**: Predict a **dense 3D face mesh** and align it with the 2D input image.
- Summary of the Connection Between P_{ST} and the 3D Face Mesh: 
	1.	Input Image → Processed by Backbone + FPN.
	2.	Feature Maps → Sent to Mesh Decoder.
	3.	Mesh Decoder → Outputs P_{ST}.
	4.	P_{ST} → Encodes shape and texture information.
	5.	3D Mesh Generation: The template mesh is deformed using graph convolutions based on P_{ST}.
	6.	Final Output: A 3D face mesh with vertices: D_{P_{ST}}.
	7.	Supervised Training: Pixel-wise L1 loss between the rendered face and input image.
- **Input**:
   - Feature maps from the **FPN**.
   - A predefined **template graph** (mesh structure) with vertices and connectivity.
- **Output**:
   - $P_{st}$: A latent 128-dimensional vector representing **shape** and **texture parameters**.
   - $ D_{P_{ST}} $: A **3D face mesh** with updated vertex positions and texture values.
- **Processing**:
   - **Mesh Decoder**:
     - Applies **graph convolutions** to predict vertex positions.
     - Outputs the 3D mesh ($ x, y, z $ coordinates) from $ P_{ST} $.
   - **Differentiable Renderer**:
     - Projects the 3D face mesh $ D_{P_{ST}} $ onto the 2D image plane.
     - Camera parameters $ P_{cam} $ and lighting $ P_{ ill} $ are applied for rendering.
- **Loss**:
   - Pixel-wise **L1 loss** between the rendered 2D face and the input face crop:
     $
     L_{ pixel} = \frac{1}{W \cdot H} \sum_{i=1}^W \sum_{j=1}^H \left\| R(D_{P_{ST}}, P_{ cam}, P_{ill})_{i,j} - I^*_{i,j} \right\|_1
     $
   - Ensures alignment of the predicted dense 3D face with the 2D input face.



### **Accuracy**
- **WIDER FACE Dataset** (Face Detection):
   - **Easy subset**: **96.9% AP** (Average Precision).
   - **Medium subset**: **96.1% AP**.
   - **Hard subset**: **91.8% AP** (best performance, outperforming the state-of-the-art).
- **Facial Landmark Accuracy**:
   - On **AFLW** dataset: Reduced normalized mean error (NME) from **2.72% to 2.21%** compared to MTCNN.
   - On **WIDER FACE validation set**: Decreased failure rate from **26.31% to 9.37%**.
- **Dense 3D Regression**:
   - On **AFLW2000-3D** dataset:
     - 68 landmarks (2D): Competitive with supervised methods like PRNet.
     - All 3D landmarks: Shows close performance to state-of-the-art supervised methods despite being self-supervised.



### **Time Efficiency**
- RetinaFace can operate in **real-time**:
   - **Heavy-weight model** (ResNet-152 backbone):
     - **13 FPS** for VGA resolution (640x480) on GPU.
   - **Light-weight model** (MobileNet-0.25 backbone):
     - **40 FPS** for 4K resolution (4096x2160) on GPU.
     - **20 FPS** for HD images (1920x1080) on CPU (multi-threaded).
     - **60 FPS** for VGA images (640x480) on single-thread CPU.
     - **16 FPS** on ARM (embedded) devices for VGA images.



### **Key Highlights**
- **Dense 3D Regression Branch**:
   - Provides self-supervised **dense 3D face alignment**.
   - Improves the robustness of face detection and alignment, especially for challenging faces.
- **Single-Stage Architecture**:
   - Similar to **YOLO**, directly predicts faces without region proposals (faster than Faster R-CNN).
- **Multi-Task Learning**:
   - Simultaneously predicts bounding boxes, landmarks, and dense 3D meshes in a single forward pass.
- **State-of-the-Art Accuracy**:
   - Best results on WIDER FACE, AFLW, and IJB-C datasets for detection, alignment, and recognition.

## SCRFD



The **SCRFD** paper (Sample and Computation Redistribution for Efficient Face Detection) focuses on improving the trade-off between **accuracy** and **efficiency** for face detection. It targets scenarios with constrained computational resources, such as real-time applications.



### 1. Overview
SCRFD introduces methods to optimize face detection by:
- **Improving inference efficiency** (reduced FLOPs, faster speed).
- Maintaining or improving detection accuracy, especially for **small faces**.



### 2. Key Methods Proposed

#### A. Sample Redistribution (SR)
- **Problem**: Small faces dominate datasets (e.g., WIDER FACE) but are underrepresented in training.
- **Solution**: A **large cropping strategy** during data augmentation:
  - Expand the random cropping range from **[0.3, 1.0]** to **[0.3, 2.0]**.
  - Generates more training samples for small faces, especially on **stride 8** feature maps.
- **Impact**: 
  - Significantly improves the detection branch responsible for small faces.

#### B. Computation Redistribution (CR)
- **Problem**: Existing models (e.g., ResNet) allocate too much computation to **deep stages** (C4, C5), which are inefficient for detecting small faces.
- **Solution**: Redistribute computation across:
  1. **Backbone stages** (Stem, C2, C3, C4, C5).
  2. **Network components** (Backbone, Neck, and Head).

##### Two-Step Strategy:
1. **Step 1**: Optimize computation within the **backbone**, favoring shallow stages (C2, C3).
2. **Step 2**: Redistribute computation across the entire network (**backbone, neck, and head**).

- **Search Process**:
   - Randomly sample model architectures under a **FLOP budget**.
   - Use **empirical bootstrap** to identify the best allocation of resources.

---

### 3. Efficiency

The SCRFD family achieves high efficiency by optimizing models for **VGA resolution (640×480)**.

#### FLOPs and Model Variants:
- **SCRFD-0.5GF**: ~0.5 billion FLOPs (mobile-level).
- **SCRFD-2.5GF**: ~2.5 billion FLOPs (low-power real-time).
- **SCRFD-10GF**: ~10 billion FLOPs.
- **SCRFD-34GF**: ~34 billion FLOPs (high-performance).

---

### 4. Accuracy

SCRFD achieves **state-of-the-art performance** on the **WIDER FACE** dataset:

| Model          | Easy (%) | Medium (%) | Hard (%) | Notes                          |
|----------------|----------|------------|----------|--------------------------------|
| **SCRFD-34GF** | 96.06    | 94.92      | 85.29    | 3× faster than TinaFace.       |
| **SCRFD-2.5GF**| 93.78    | 92.16      | 77.87    | Significant improvement.       |
| **SCRFD-0.5GF**| 90.57    | 88.12      | 68.51    | Outperforms RetinaFace-Mobile. |

- **Comparison**:
  - **SCRFD-34GF** outperforms **TinaFace** by **3.86% AP** on the hard subset while being **3× faster**.

---

### 5. Time Complexity (Inference Speed)

SCRFD is tested at **VGA resolution** with single-scale testing:

| Model          | FLOPs (G) | Inference Time (ms) | Device          |
|----------------|-----------|---------------------|-----------------|
| **SCRFD-34GF** | 34.13     | 11.7                | NVIDIA 2080Ti   |
| **SCRFD-2.5GF**| 2.53      | 4.2                 | NVIDIA 2080Ti   |
| **SCRFD-0.5GF**| 0.508     | 3.6                 | NVIDIA 2080Ti   |
| **TinaFace**   | 172.95    | 38.9                | NVIDIA 2080Ti   |

---

### 6. Key Contributions

1. **Sample Redistribution**:
   - Improves training for **small faces** by increasing positive samples at small scales.
2. **Computation Redistribution**:
   - Optimizes the allocation of computation across backbone, neck, and head.
3. **Efficiency**:
   - Delivers high accuracy with significantly lower computational cost and inference time.
4. **State-of-the-Art Results**:
   - Outperforms competing face detectors (TinaFace, HAMBox, RetinaFace, DSFD) across multiple compute regimes.

---

### 7. Practical Significance

- **Real-Time Applications**:
   - Suitable for deployment on low-latency systems like mobile devices and edge AI.
- **Scalability**:
   - The SCRFD family provides optimized models across a range of compute budgets:
     - **Low-end**: SCRFD-0.5GF (mobile-level).
     - **Mid-range**: SCRFD-2.5GF.
     - **High-end**: SCRFD-34GF.
- **Deployment-Friendly**:
   - Optimized for **single-scale testing**, avoiding the overhead of multi-scale testing.

---

### 8. Key Takeaways

- **Why SCRFD?**
   - Efficient, accurate, and scalable for face detection tasks.
   - Optimized for detecting **small faces**, critical in real-world datasets.
- **Performance**:
   - High AP (Average Precision) with low computational cost (FLOPs) and fast inference time.
   - Best-in-class efficiency for resource-constrained environments.

---

### 9. Performance Summary

| Model          | FLOPs (G) | Hard AP (%) | Inference Time (ms) |
|----------------|-----------|-------------|---------------------|
| **SCRFD-34GF** | 34.13     | 85.29       | 11.7                |
| **SCRFD-2.5GF**| 2.53      | 77.87       | 4.2                 |
| **SCRFD-0.5GF**| 0.508     | 68.51       | 3.6                 |
| **TinaFace**   | 172.95    | 81.43       | 38.9                |

---

### 10. Conclusion

The SCRFD paper presents a highly efficient and scalable face detection framework that achieves:
- **Optimized computation** through Sample and Computation Redistribution.
- **State-of-the-art accuracy** with significantly reduced computational cost.
- **Real-time performance** across a wide range of devices and compute constraints.

SCRFD is ideal for both **resource-constrained environments** and **high-performance real-time face detection**.

## BlazeFace 

**Main Purpose**  
- Develop a lightweight, fast, and accurate face detection model optimized for real-time performance on mobile devices.  
- Achieve sub-millisecond inference times on mobile GPUs while maintaining competitive detection accuracy.

**Method**  
- Utilize a single-shot detection paradigm (similar to SSD) for direct face bounding box regression and classification.  
- Employ a streamlined neural network architecture with depthwise-separable convolutions to reduce computational complexity.  
- Carefully design anchor boxes and feature maps to efficiently handle multiple face scales.  
- Optimize the network for mobile GPUs by selecting operations and layer dimensions that run efficiently on mobile hardware.

**Results**  
- Demonstrates strong detection performance on standard face datasets.  
- Achieves near state-of-the-art accuracy for face detection tasks compared to more computationally expensive models.  
- Compact and efficient enough to run on-device for applications like video calling, AR filters, and camera interfaces.

**Accuracy**  
- Comparable accuracy to more complex face detection architectures while being significantly smaller and faster.  
- Robust detection across various lighting conditions, face sizes, and angles, aided by effective data augmentation and anchor strategy.

**Performance**  
- Highly optimized for mobile environments, requiring less memory and fewer computational resources.  
- Maintains stable frame rates for real-time video processing, ensuring a smooth user experience.

**Time Efficiency**  
- Achieves sub-millisecond inference times on devices like the Google Pixel 2’s GPU.  
- Eliminates face detection as a latency bottleneck, enhancing responsiveness in live video streaming and AR applications.

<br><br>

# dlib

It is a C++ toolkit provides several machine learning solutions. It has python API. 
-  [Here](http://dlib.net/dnn_yolo_train_ex.cpp.html) is an example illustrating the use of the deep learning tools from the dlib C++
    Library.  In this example program it is showed how one can train a YOLO detector.

- [*get_frontal_face_detector*](http://dlib.net/dlib/image_processing/frontal_face_detector_abstract.h.html#get_frontal_face_detector)  is configured to find human faces that are looking more or less towards the camera. The tool is based on sliding a linear classifier over a HOG pyramid( [*scan_fhog_pyramid*](http://dlib.net/dlib/image_processing/scan_fhog_pyramid_abstract.h.html#scan_fhog_pyramid)).

- [Here](http://dlib.net/face_landmark_detection.py.html) is a python example showing how to find frontal human faces in an image and estimate their pose.  The pose takes the form of 68 landmarks.

-  [This](http://dlib.net/face_landmark_detection_ex.cpp.html) example program shows how to find frontal human faces in an image and estimate their pose.  The pose takes the form of 68 landmarks. The face detector used is made using the classic Histogram of Oriented Gradients (HOG) feature combined with a linear classifier, an image pyramid, and sliding window detection scheme.  

- The above tools are based on classical algorithms. If some deep learning implementations is needed for fundamental tasks in object detection, follow this [link](http://dlib.net/ml.html#loss_mmod_). The deep learning approach for object detection is based on [*Max Margin Object Detection*](https://arxiv.org/abs/1502.00046) method.

<br><br>
# YOLO5FACE paper (2022)

## **1. YOLO5Face Model Versions and Naming Convention**

### **Naming Convention**
The naming convention for YOLO5Face models:
- **Base name**: `YOLOv5` + size indicator (`s`, `m`, `l`, `x`, etc.).
- **Suffix `6`**: Indicates that the model includes the **P6 output block**, enhancing detection of large faces.
- **Small models**: Use lightweight backbones like **ShuffleNetV2** for embedded devices.

| **Component**       | **Meaning**                                                                                         |
|---------------------|-----------------------------------------------------------------------------------------------------|
| **s, m, l, x**      | Indicates model size: small (`s`), medium (`m`), large (`l`), extra-large (`x`).                    |
| **6**               | Adds P6 output layer for better large face detection.                                               |
| **ShuffleNetV2**    | Lightweight backbone for small models (`YOLOv5n`).                                                  |
| **0.5**             | Further scaled-down ShuffleNetV2 for extremely lightweight cases (`YOLOv5n0.5`).                    |

---

## 2. YOLO5Face Performance Metrics on WIDERFACE VALIDATION DATASET

### **Larger Models (CSPNet Backbone)**

| **Model**          | **Backbone**          | **P6?** | **Params (M)** | **FLOPS (G)** | **Easy mAP (%)** | **Medium mAP (%)** | **Hard mAP (%)** |
|---------------------|-----------------------|---------|----------------|---------------|------------------|--------------------|------------------|
| **YOLOv5s**        | YOLOv5-CSPNet        | No      | 7.075          | 5.751         | 94.33            | 92.61              | 83.15            |
| **YOLOv5s6**       | YOLOv5-CSPNet        | Yes     | 12.386         | 6.280         | 95.48            | 93.66              | 82.80            |
| **YOLOv5m**        | YOLOv5-CSPNet        | No      | 21.063         | 18.146        | 95.30            | 93.76              | 85.28            |
| **YOLOv5m6**       | YOLOv5-CSPNet        | Yes     | 35.485         | 19.773        | 95.66            | 94.10              | 85.20            |
| **YOLOv5l**        | YOLOv5-CSPNet        | No      | 46.627         | 41.607        | 95.90            | 94.40              | 84.50            |
| **YOLOv5l6**       | YOLOv5-CSPNet        | Yes     | 76.674         | 45.279        | 96.38            | 94.90              | 85.88            |
| **YOLOv5x6**       | YOLOv5-CSPNet        | Yes     | 141.158        | 88.665        | 96.67            | 95.08              | 86.55            |

### **Lightweight Models (ShuffleNetV2 Backbone)**

| **Model**          | **Backbone**           | **P6?** | **Params (M)** | **FLOPS (G)** | **Easy mAP (%)** | **Medium mAP (%)** | **Hard mAP (%)** |
|---------------------|------------------------|---------|----------------|---------------|------------------|--------------------|------------------|
| **YOLOv5n**        | ShuffleNetV2          | No      | 1.726          | 2.111         | 93.61            | 91.54              | 80.53            |
| **YOLOv5n0.5**     | ShuffleNetV2-0.5      | No      | 0.447          | 0.571         | 90.76            | 88.12              | 73.82            |

---

## **3. Comparison with RetinaFace and SCRFD**

### **Evaluation Dataset**
The models were evaluated on the **WiderFace dataset**, which is split into:
- **Easy Subset**: Images with minimal challenges (e.g., clear faces).
- **Medium Subset**: Images with moderate challenges.
- **Hard Subset**: Images with extreme challenges (e.g., small, occluded, or poorly lit faces).

---

### **Performance Comparison on WiderFace**

| **Model**           | **Params (M)** | **FLOPS (G)** | **Easy mAP (%)** | **Medium mAP (%)** | **Hard mAP (%)** |
|----------------------|----------------|---------------|------------------|--------------------|------------------|
| **YOLOv5x6-Face**   | 141.158        | 88.665        | 96.67            | 95.08              | 86.55            |
| **YOLOv5l6-Face**   | 76.674         | 45.279        | 96.38            | 94.90              | 85.88            |
| **SCRFD-34GF**      | 51.3           | 34.0          | 96.52            | 94.99              | 85.21            |
| **SCRFD-10GF**      | 11.7           | 10.0          | 95.55            | 93.67              | 81.40            |
| **RetinaFace**      | 29.3           | 13.1          | 95.96            | 93.58              | 81.57            |

---

### **Key Observations**
1. **YOLO5Face Advantages**:
   - **YOLOv5x6-Face** achieves the highest mAP scores on all subsets:
     - **Easy**: 96.67%
     - **Medium**: 95.08%
     - **Hard**: 86.55%
   - Consistently outperforms SCRFD-34GF and RetinaFace on all subsets, especially for the most challenging (Hard) subset.

2. **Efficiency Comparison**:
   - YOLO5Face models with P6 (e.g., YOLOv5l6-Face, YOLOv5x6-Face) deliver higher accuracy than SCRFD and RetinaFace while maintaining comparable computational complexity.
   - Lightweight YOLO5Face models (e.g., YOLOv5n-Face) provide competitive performance with significantly fewer parameters and FLOPS.

3. **Performance on Challenging Faces**:
   - YOLO5Face's inclusion of the **P6 output block** and **custom data augmentation** improves performance on small and occluded faces, outperforming both SCRFD and RetinaFace in the Hard subset.

---

## **4. Summary Table**

| **Model**           | **Params (M)** | **FLOPS (G)** | **Best Feature**                                   |
|----------------------|----------------|---------------|--------------------------------------------------|
| **YOLOv5x6-Face**   | 141.158        | 88.665        | Best accuracy across all WiderFace subsets.      |
| **YOLOv5n0.5-Face** | 0.447          | 0.571         | Lightweight, suitable for embedded systems.      |
| **SCRFD-34GF**      | 51.3           | 34.0          | High accuracy with moderate computational cost.  |
| **RetinaFace**      | 29.3           | 13.1          | Good balance of accuracy and FLOPS.              |

---

## **5. Conclusions**

- **YOLO5Face Strengths**:
  - Superior performance in **mAP scores** compared to RetinaFace and SCRFD.
  - Scalable models for diverse use cases: from **lightweight models** (YOLOv5n) to **high-performance models** (YOLOv5x6).
  - Enhanced detection of small, occluded, and large faces due to **multi-scale detection** and the **P6 output block**.

- **When to Choose YOLO5Face**:
  - **Real-time applications**: Use YOLOv5n0.5 or YOLOv5n.
  - **High-accuracy applications**: Use YOLOv5x6 or YOLOv5l6.

## YOLO5Face: Brief Description

**YOLO5Face** is a face detection model based on the YOLOv5 object detection framework, optimized for accurate, scalable, and efficient face detection. Below is a concise summary of its key features and contributions:

---

## **Key Features**
1. **Enhanced Architecture**:
   - Built on the YOLOv5 framework with modifications for face detection.
   - Adds a **5-point facial landmark regression head** to predict key facial points (eyes, nose, mouth corners).
   - Includes a **P6 output block** to improve detection of large faces.

2. **Backbone and Variants**:
   - Utilizes **CSPNet** for high-performance models.
   - Lightweight models leverage **ShuffleNetV2** for mobile and embedded applications.
   - Variants include `YOLOv5s`, `YOLOv5m`, `YOLOv5l`, and `YOLOv5x` with optional `6` suffix (e.g., `YOLOv5x6`).

3. **Data Augmentation**:
   - Custom data augmentation techniques enhance robustness.
   - Avoids up-down flipping and over-reliance on Mosaic augmentation.

4. **Loss Functions**:
   - Uses **CIoU Loss** for bounding boxes and **Wing Loss** for landmark regression, improving accuracy, particularly for small errors.

---

## **Performance Highlights**
1. **State-of-the-Art Results**:
   - Achieves top performance on the **WiderFace dataset**:
     - **Easy mAP**: 96.67%
     - **Medium mAP**: 95.08%
     - **Hard mAP**: 86.55%

2. **Efficiency**:
   - Real-time detection across all models.
   - Lightweight models (e.g., `YOLOv5n`) optimized for embedded devices with minimal computational requirements.

3. **Robust Detection**:
   - Excels at detecting small, occluded, and large faces due to multi-scale detection and the P6 output block.

---

## **Comparison with Competitors**
1. **Better Accuracy**:
   - Outperforms **RetinaFace** and **SCRFD** on WiderFace, especially on the Hard subset.

2. **Scalable Models**:
   - Offers flexibility with both high-performance and lightweight variants.

---

## **Use Cases**
- Real-time face detection for **embedded systems**, **mobile applications**, and **high-performance systems**.
- Integration with downstream tasks like **face recognition** due to its precise landmark regression.

---

## **Conclusion**
YOLO5Face is a versatile and efficient face detection model that achieves state-of-the-art accuracy with scalable performance for a wide range of applications.

## Summary of the Discussion on YOLOv1 Training and Concepts

Below is a concise overview of the main points covered in our conversation about YOLO (specifically YOLOv1) and its training procedure.

---

### 1. YOLOv1 Basics

- **Grid Division**: YOLOv1 divides the input image into an \(S * S\) grid (commonly 7×7).
- **Bounding Boxes per Cell**: Each cell predicts 2 bounding boxes; these are not “anchor boxes.” Instead, each bounding box is free-form, with \(x, y, w, h\) learned from scratch.
- **Single Responsibility**:
  - Only one cell (the one whose center lies inside a ground-truth object’s box) is responsible for predicting that object.
  - Among the 2 predicted boxes in that cell, the one with the highest IoU to the ground truth is assigned to predict that object.

---

### 2. Label Assignment and Loss

- **Labels**:
  1. **Coordinates** (x, y, w, h): Regressed directly, typically normalized to [0, 1].
  2. **Confidence**: Set to the IoU between the predicted box and ground truth (for the “responsible” box). Zero for other boxes that do not match any object.
  3. **Class Probabilities**: One-hot vector of the object’s class for the responsible cell. The entire cell shares one class distribution in YOLOv1.
- **Loss Function**: Sum of squared errors on:
  1. Box coordinates  
  2. Box confidence  
  3. Class probabilities  


---

### 3. No Anchors in YOLOv1

- Unlike YOLOv2 and later versions, YOLOv1 has **no pre-defined anchor boxes**. 
- Each cell’s 2 bounding boxes are entirely learned.  
- This differs from anchor-based methods where bounding box predictions are offsets relative to fixed template boxes (anchors).

---

### 4. Forward Pass and IoU Computation

1. **Forward Pass**:
   - The network outputs (x, y, w, h), confidence, and class probs for each cell.
2. **IoU Calculation**:
   - Even though there are no anchors, the predicted boxes are still in image space (after decoding (x, y, w, h) into absolute coordinates).
   - We can directly compute IoU between each predicted box and the ground-truth box.
3. **Assignment**:
   - Whichever bounding box has the highest IoU with the ground truth is chosen as “responsible.”

---

### 5. Training Workflow

1. **Load an image** and **annotations** (\(x, y, w, h\) + class).
2. **Determine which cell** is responsible for each ground-truth box (the cell containing the box’s center).
3. **Compute IoU** for that cell’s 2 box predictions vs. the ground truth.
4. **Pick the best-matching box**; label it to predict \((x, y, w, h)\) and confidence = IoU, and set the class label for that cell.
5. **Other boxes** in the grid: confidence = 0 if not responsible for any object.
6. **Loss computation** (coordinates, confidence, class probabilities).
7. **Backpropagation** to update the model weights.
8. **Repeat** for each training sample or batch over multiple epochs.

---

**In conclusion**, YOLOv1 uses a grid-based approach where each cell regresses bounding boxes from scratch (no anchor templates) and only one cell is responsible for each object. Confidence is learned to match IoU, and class predictions are per cell. This mechanism was foundational for later YOLO variants, which introduced anchor boxes and multi-scale detection to improve performance on diverse object sizes.

## Differences Between YOLO5Face and SSD Framework

Here’s a concise overview of the key differences between YOLO5Face (based on YOLO) and SSD (Single Shot MultiBox Detector):

---

### **1. Grid Cell Responsibility**
- **YOLO5Face**:
  - Each object is assigned to a **single grid cell** based on the center of the object.
  - Within the grid cell, the **best matching anchor box** is chosen using IoU.
- **SSD**:
  - Objects can be assigned to **multiple grid cells**.
  - An object is matched to all anchor boxes with IoU exceeding a threshold (e.g., 0.5).

---

### **2. Anchor Box Matching**
- **YOLO5Face**:
  - Matches **one anchor box per object** (best IoU) within the responsible grid cell.
- **SSD**:
  - Matches **multiple anchor boxes per object**, even across different grid cells and feature maps.

---

### **3. Detection Layers**
- **YOLO5Face**:
  - Uses **fewer detection layers** (e.g., P3, P4, P5, and optionally P6).
  - The P6 layer improves detection of large faces.
- **SSD**:
  - Uses **more detection layers** across feature maps (e.g., 38 * 38, 19 * 19, 10 * 10), ensuring better coverage for objects of all sizes.

---

### **4. Negative Sampling**
- **YOLO5Face**:
  - Negative samples are treated implicitly (grid cells or anchor boxes not matched to any object).
- **SSD**:
  - Employs **hard negative mining** to focus on the most challenging negative samples.

---

### **5. Training Complexity**
- **YOLO5Face**:
  - Simpler training process, with each object assigned to a single anchor box.
- **SSD**:
  - More complex training due to the many-to-many matching of ground truth boxes and anchor boxes.

---

### **6. Inference Efficiency**
- **YOLO5Face**:
  - Faster due to fewer predictions per grid cell and a single responsible anchor box per object.
- **SSD**:
  - Slower inference as it generates more redundant predictions, which require additional filtering (e.g., Non-Maximum Suppression).

---

### **Summary Table**

| **Aspect**                | **YOLO5Face**                          | **SSD**                                  |
|---------------------------|----------------------------------------|------------------------------------------|
| **Grid Responsibility**   | One grid cell per object               | Multiple grid cells per object           |
| **Anchor Matching**        | Single best IoU anchor per object      | Multiple anchors per object              |
| **Detection Layers**       | Few layers, P6 for large faces         | Many feature maps for multi-scale detection |
| **Negative Sampling**      | Implicit                              | Hard negative mining                     |
| **Training Complexity**    | Simpler                               | More complex                             |
| **Inference Efficiency**   | Faster                                | Slower due to redundant predictions      |

---

YOLO5Face prioritizes **efficiency** and **simplicity**, while SSD focuses on **flexibility** and **recall** through many-to-many matching and dense feature maps.


<br> <br>

# EfficientFace Paper (2023)

EfficientFace is a lightweight face detection framework designed to balance **accuracy** and **efficiency** for real-time applications. The model aims to overcome the limitations of existing lightweight face detectors, which often sacrifice accuracy for efficiency, by incorporating feature enhancement mechanisms.

---

## 1. Overview of EfficientFace
EfficientFace introduces key modules to enhance feature representation while maintaining computational efficiency.

### Key Contributions
1. **SBiFPN (Symmetrically Bi-directional Feature Pyramid Network):**
   - Facilitates cross-scale feature fusion by shortening the pathway between low- and high-level features.
   - Ensures the fusion of both semantic and localization information.

2. **Receptive Field Enhancement (RFE):**
   - Addresses the challenge of detecting faces with unbalanced aspect ratios by creating features with diverse receptive fields.

3. **Attention Mechanism (AM):**
   - Combines spatial and channel attention to enhance the representation of occluded faces.

4. **High Efficiency and Competitive Accuracy:**
   - Achieves a balance between performance and computational cost, leveraging the EfficientNet backbone and feature enhancement modules.

---

## 2. Architecture Details
- **Backbone:**
  - EfficientFace uses **EfficientNet-B5** for feature extraction.
  - Feature maps are extracted from six levels: C2, C3, C4, C5 (backbone layers), and C6, C7 (downsampled layers).

- **Feature Enhancement Modules:**
  - **SBiFPN:**
    - A bi-directional, parallel, symmetrical feature pyramid network that fuses multi-scale features in a single iteration.
    - Employs a weighted fusion strategy to aggregate features efficiently.
  - **RFE:**
    - Utilizes convolutional layers of varying kernel shapes (e.g., 1×5, 5×1, etc.) to create features that can handle faces with diverse aspect ratios.
  - **Attention Mechanism:**
    - Combines spatial and channel attention to improve detection of occluded faces.
    - Optimized with a depth of 2 (repeated twice for best performance).

- **Prediction Heads:**
  - Six heads (one per feature level: C2 to C7) for face classification and bounding box regression.
  - Anchors are predefined at scales: `{16, 32, 64, 128, 256, 512}`.

---

## 3. Performance

### Accuracy
The performance of EfficientFace was evaluated on four datasets: **WIDER Face**, **AFW**, **PASCAL Face**, and **FDDB**. On the WIDER Face dataset (validation set), the model achieves:
- **Easy subset**: 95.1%
- **Medium subset**: 94.0%
- **Hard subset**: 90.1%

On other datasets:
- **AFW**: 99.94% (AP)
- **PASCAL Face**: 99.38% (AP)
- **FDDB**: 97.0% (True Positive Rate @ 1,000 False Positives)

### Comparison with State-of-the-Art
- EfficientFace is competitive with heavyweight models like DSFD and SRNFace-2100 while having significantly lower computational costs.
- Outperforms lightweight models (e.g., YOLOv5n, EXTD) by a significant margin in accuracy.

---

## 4. Efficiency

### Model Parameters
- EfficientFace has **31.46 million parameters**, significantly fewer than heavyweight models like DSFD (120M) and MogFace (85M+).

### MACs (Multiply-Accumulate Operations)
- **52.59 G MACs**, making it 6.5× more efficient than DSFD (345.16 G MACs) and 4.8× more efficient than SRNFace-2100 (251.94 G MACs).

### Runtime Performance
- The paper does not explicitly declare the **inference time** or FPS (frames per second), but the stated efficiency metrics indicate its suitability for real-time applications.

---

## 5. Time and Memory Complexity

### Time Complexity
- **Not explicitly stated in the paper.**
- However, the use of EfficientNet-B5 and the single-iteration SBiFPN suggests reduced computational time compared to iterative fusion methods like BiFPN.

### Memory Complexity
- **Not explicitly stated in the paper.**
- The reduced parameter count and use of lightweight modules imply lower memory requirements compared to heavyweight detectors.

---

## 6. Training and Implementation Details
- Pre-trained on the **COCO dataset** using EfficientNet.
- Optimized with the AdamW optimizer and ReduceLROnPlateau learning rate schedule.
- Training configuration:
  - Initial learning rate: `10^{-4}`
  - Batch size: 4
  - Maximum channels: 288
- Experiments conducted on an **NVIDIA GTX 3090** GPU using the PyTorch framework.

---

## 7. Key Observations

### Strengths
1. EfficientFace achieves a **favorable balance between accuracy and efficiency**, outperforming most lightweight models while remaining competitive with heavyweight models.
2. The novel **SBiFPN** module significantly enhances feature fusion and ensures accurate detection across scales.
3. **RFE and attention mechanisms** improve the model’s ability to detect faces with extreme aspect ratios and occlusions.

### Limitations
1. The paper does not provide direct benchmarks for inference speed (e.g., FPS) or memory usage.
2. The computational complexity is only expressed in terms of MACs, which might not fully represent real-world runtime performance.

---

## 8. Summary
EfficientFace is a well-rounded, lightweight face detector that leverages advanced feature fusion (SBiFPN), receptive field diversity (RFE), and attention mechanisms to deliver state-of-the-art accuracy and efficiency. It is particularly effective in detecting faces across challenging conditions like occlusion and extreme aspect ratios while maintaining computational efficiency.

## 9. EfficientNet as backbone: 

### MBConv in EfficientNet

- **Definition**: MBConv (Mobile Inverted Bottleneck Convolution) is an efficient convolutional block used in EfficientNet, originating from MobileNetV2.

---

### **Key Features**
- **Inverted Residual**:
  - Expands the input to higher dimensions, applies depthwise convolution, then compresses back.
  - Reverse of traditional bottleneck structure.

- **Components**:
  1. **Expansion Phase**: Expands feature maps using a 1x1 convolution (e.g., 6x the input channels).
  2. **Depthwise Convolution**: Applies channel-wise convolution for computational efficiency.
  3. **Pointwise Compression**: Reduces channels back using a 1x1 convolution.
  4. **Squeeze-and-Excitation (SE)**: Weighs channel importance to improve feature representation.
  5. **Skip Connection**: Bypasses the block when input and output shapes match.

---

### **Advantages**
- **Reduced Parameters**: Depthwise convolutions minimize the number of parameters.
- **Efficient Computation**: Operates effectively in a compressed feature space.
- **Enhanced Features**: Expansion and SE modules improve feature expressiveness.

---

### **Role in EfficientNet**
- **Primary Building Block**: Used throughout EfficientNet for feature extraction.
- **Flexible Scaling**: Configured with different expansion factors, kernel sizes, and strides across variants (B0 to B7).

---

### **Summary**
MBConv optimizes the trade-off between accuracy and computational cost, enabling EfficientNet to perform well with fewer resources.

<br><br>

# OpenVino

- **OpenVINO™ (Open Visual Inference and Neural Network Optimization):**
  - Open-source toolkit developed by Intel®.
  - Optimizes and deploys deep learning models across various hardware platforms.
  - Enhances AI inference performance with tools for model optimization, conversion, and deployment.
  - [Learn more on Intel's website](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html?utm_source=chatgpt.com).



### **Key Features**
- **Inference Optimization**:
  - Boosts performance for tasks like computer vision, automatic speech recognition, generative AI, and NLP.
  - [Explore more on GitHub](https://github.com/openvinotoolkit/openvino?utm_source=chatgpt.com).

- **Flexible Model Support**:
  - Compatible with models from frameworks like PyTorch, TensorFlow, ONNX, Keras, PaddlePaddle, and JAX/Flax.
  - Direct integration of models from sources like the Hugging Face Hub using Optimum Intel.

- **Broad Platform Compatibility**:
  - Efficiently deploys on platforms ranging from edge to cloud.
  - Supports inference on:
    - CPUs (x86, ARM).
    - GPUs (OpenCL capable, integrated, and discrete).
    - AI accelerators (Intel NPU).



### **Workflow Overview**
1. **Model Acquisition**:
   - Obtain a pre-trained model from supported frameworks.
   
2. **Model Conversion**:
   - Use the OpenVINO Model Optimizer to convert the model into an Intermediate Representation (IR) consisting of:
     - `.xml` file (model architecture).
     - `.bin` file (model weights).

3. **Model Optimization**:
   - Apply techniques like quantization or pruning using tools like the Neural Network Compression Framework (NNCF).

4. **Inference Deployment**:
   - Deploy the optimized model using the OpenVINO Runtime.
   - Specify the target inference hardware (e.g., CPU, GPU, VPU).



### **Additional Information**
- **Cross-Platform**:
  - Supports Windows, Linux, and macOS.
  - Provides APIs in C++, Python, and Node.js.

- **License**:
  - Available under the Apache License 2.0.
  - Free for personal and commercial use.

- [Official Documentation](https://docs.openvino.ai/?utm_source=chatgpt.com)

<br><br>

# ULTRA-LIGHT github repo

- The code for detecting faces from a set of images or a fron video webcame is available
- The model that is working now is `slim-320-quant-ADMM-50.mnn` the other two onces did not work due to some problems with the MNN framework or something else. 
- I want to improve the temporal jitter for real time when using front webcame in real time but I think it is not important that much. the method i wanted to use is EMA, it is implemented by chatGPT. 
- The problem is that these models are in `.mnn` format which is not supported by ST platform to be deployed on the MCUs. To this end, I am gonna drop work as of now. 
- [Link to the MNN doc](https://www.yuque.com/mnn/en/usage_in_python#6wFFD)
- [Link to the github repo of the models](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/MNN)

<br><br>

# ONNX model zoo (Ultra-lightweight face dectector)

- [Github link](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/ultraface)
- The model `version-RFB-320-int8.onnx` is the `onnx` version of the model in the **ULTRA-LIGHT** source.
- version-RFB-320-int8 is obtained by quantizing fp32 version-RFB-320 model. We use [Intel® Neural Compressor](https://github.com/intel/neural-compressor) with onnxruntime backend to perform quantization. View the [instructions](https://github.com/intel/neural-compressor/blob/master/examples/onnxrt/body_analysis/onnx_model_zoo/ultraface/quantization/ptq_static/README.md) to understand how to use Intel® Neural Compressor for quantization.
- There are three models and all of them are working
 > NOT IMPLEMENTED: Onnx exporting model with quantized unsigned integer format is not supported. This is the error for in8 Model
 > NOT IMPLEMENTED: Order of dimensions of input cannot be interpreted. This is the erro for other models 




<br><br>

# PINTO MODEL ZOO

## DSFD Vgg 

- [Link to the repo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/040_DSFD_vgg)
- There are only two links in this repo. One of the links contains some models and there is weight quantized model in there
- The other link is [another github](https://github.com/610265158/DSFD-tensorflow/tree/master) repo that implements dsdf method in tensorflow and also contains the `.pd` files of the model
- In the `/Models/` folder, the both lightnet models are tested and they worked, how ever the models in in the folder `/OtherVariations/` did not worked.
- There is a face dataset `.npy` in the models found folder in the related forlder

## DBFace

- [Link to the repo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/041_DBFace)
- All the models worked properly, however, there are two code files for inference based on the model. see the main block of each of the files. 

## Face Landmark

- It is working but it is so slow, because there are two networks in pipline and running on CPU is not efficient.
- there is a `YOLOV4` model provided in this repo for head detection. which is a huge model. 
- [Link to chapGPT chat about this work.](https://chatgpt.com/share/67aa15b2-0a9c-8011-9ee5-af12e373b8c3)
- There are INT8 and other variations of the landmark model is provided, check the FoundModels folder.
- both the networks for head detection and face landmark detection that are working are `onnx` format.
- Both of these `onnx` models are tested on `STM32EdgeAI` platform as well as one other face landmark `tflite` model.
- there are several tflite models for face landmark model. A new python program for working with tflite face landmark model is developed but the final dense face mesh is not working. Run the code for more information.

## Iris Landmark

- It was not mentioned anywhere how to interpret the outputs of the models related to this repository. 
- There was no code for doing inference on input image.
- I just developed a code for doing infrence on input video stream, however, it is not working because I do not know how to interpret the outputs
- For sure the model must be coupled with a face detector
- I have not tested any of these models on `STM32EdgeAI`. 

## CenterFace

- The model is working and the input is of any size.
- A model with post processing embedding in it is also available but the program for inference works with original model. 


## RetinaFace

- did not go through it because the download file for it was so heavy and also no explicit code for inference was provided.
- Apparently, the models provided for this repo are based on ResNet-50 which is not suitable for out usecase.

## WHENet

- Coupled the model with `YOLOV4` for real time inference but it was so slow and also the model does not seem stable and robust. 
- Tested two version of the model. 
- ***WHENet is for head pose estimation.*** 

## SCRFD 

- There are two set of codes available for inference from `onnx` or `tflite` models. 
- The code for `onnx` models are working.
- Some versions of the model return landmarks (`bnkps`) while other versions do not.
- Looking at table, the smallest model that was compatible with the target device is evaluated but the RAM size apprantely is not applicable so assessing other models is dropped.
- `onnx` models are not working on the ST platform. `tflite` models are working but apparentely the smallest one is still bigger than the memory requirements. 


## Head Pose Estimation Adas-0001

- This model outputs three numbers that I think are related to three angles of a head. 
- I should be coupled with a face detector for real applications. I did not run the models. 
- [official repo for adas head pose estimation](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/head-pose-estimation-adas-0001) -> some refrences for gaze estimations as well are available
- I think the demo code used in the PINTO model zoo or the official repo are working with openvino models. 
- The three version of the models (onnx, openvion, tflite) are evaluated on BIWI test set.

## YuNet

- The non quantized model is working it performs well and also gives us the landmarks. 
- The quantized ones did not work there are a lot of confliction when running the code for inference. 
- we can use the quantized model for testing on STM32 edge AI application and if it goes well, then we can debug the code for inferece for them.
- The model is performing good in term of jitter.
- there is also available the version 2 of this model in this PINTO [link](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/387_YuNetV2)
- The models resources zip file is not downloaded, there is a link to another git repo that contains some models. These models are only considered. 
- there are two set of models, two of the return landmarks that are compatible with the inference code, the other two do not return landmarks and are not compatible with the inference code. Actually the outputs of them are ambigious 
- There is only one other `onnx` models that is not compatible on the target device, see the table. 


## Face Detection Adas-0001

- there is not code provided for inference
- I have downloaded the models, could not understand the output of the model architecture.
- The model `model_float32.onnx` is tested on the ST platform and it gives the following results: MACC = 1,418,947,620 / Flash = 4.1 MB / RAM: 33.3. Therefore, since the RAM is to much i did not included it in the table. 
- At this point, i realized that using the random quatization option on the platform significantly reduce the model size. So if we decided to evaluate only quantized models we can take this model into account.

## Lightweight Pose Estimation

- A lightweight model for headpose estimation

## 6DRepNet

- this seems to be good model for head pose 3d estimation, I should study it further.

## SLPT

- This is a model for predicting dense face mesh, needs to be coupled with a face detector. 
- Skipping it for now.
- Seems to be a big model.




## FAN

- It is a model for predicting 68 face landmark and must be coupled with a face detector.
- Skipped 
- demo inference code provided.


## SynergyNet

- 3D facial alignment, face orientation estimation, and 3D face modeling.
- must be coupled with a face detector.
- No code provided for inference.
- Skipped

## DMHead

- Dual model head pose estimation. Fusion of SOTA models. 360° 6D HeadPose detection. All pre-processing and post-processing are fused together, allowing end-to-end processing in a single inference.
- No inference code is provided
- Skipped

## HHP Net

- A simple neural network to estimate the head pose of people in single images starting from a small set of head keypoints.
- it should be coupled with a face detector and a keypoints detector. 
- No demo code available. 
- Skipped

## YOLOv7 Head

- All the models work well. 
```diff
+ An original models was not working on the software but the random quantized worked. 
```

## DirectMHP

- This is an amazing model for face detection and head pose estimation that is single stage network
- The models in the code base folder are working, but one of them did not
- There are a lot of variations of the models available
- the models in the code base include the post processing inside the model architecture
- there is the paper of this model
- some parameters of the models like IOU or thresholds maybe part of the post-processing embedded in the model architecture, so playing with these parametrs may not be trivial. There is tutorial on how to change them in the PINTO repo.
```diff
+ Since the post-processing is inside the model, it is not possible to deploy it on the device
```
## RetinaFace MobileNetV2

- A model works perfectly and very good.
- Again, like the previous case there are two set of models available that one of them has the post-processing embedded in the model architecture while the other one is original model.
- there are several variations of models with post-processing with different thresholds or other parameters.
```diff
+ The models with post processing are not applicable on the target device
```

## FaceMeshV2

- There is no code for inference. So it is needed to develope a code for inferece.
- There are a few versions available.


## STAR 

- for landmark detection 
- two stage model that uses YOLO for face detection

```diff
+ This is an accurate network for predicting 2-d face mesh (landmark)
+ it needs to be coupled with a face detector 
+ In the inference programm it is coupled with a version of the 'YOLO X' model that is so heavy so that the final inference time is so slow. 
+ The STAR models is tested with the platform but this error is obtained : " INTERNAL ERROR: Error in output shape computation of {}: number of output shapes and output values are different" 
```

## 6DRepNet360

- This is a netwrok for head pose estimation so it must be coupled with a head detector 
- Uses YOLO for detection.
- There is an inference code working good for head pose estimation
- There are other codes for inference that are for gaze tracking and ... but they need a 3d party library that it is not available.

```diff
+ We can use for gaze tracking and head monitoring.
+ There is "gold_yolo_n_head_post_0277_0.5071_1x3x480x640.onnx" in this network that includes post processing in the network and it works and also it is not applicable on the target device. 
+ Also, the 6drepnet model is not applicable on the device. 
```

## FaceBox

- The models seems so robust and fast. 
- The inference program is for models with post processing inside model. 

## MobileFaceNet

- For landmark detection (almost dense face mesh)
- Two stage network
- Like `STAR` but not better
- The model is tested on the platform and it worked but it was facing an error when getting the random quantized version on the platform. 


## Peppa Pig Face Landmark

- This is a network like `MobileFaceNet` and the `STAR` for face mesh detection.
- The model is tested on the platform but an error raised due to not implemented layers, also for quantizing the model on the platform.
- The code for inference is available. 
- The model size seems big.
- There are student-teacher versions of the model. 



## PipNet 

- Exactly Same as previous case
- The program for inference is available

## Opal23 Head Pose

- Same scenario as previous models however this model is for detectiong head pose.
- The model cannot be deployed on the device : `INTERNAL ERROR: Error in output shape computation of {}: number of output shapes and output values are different`
- And also, the model cannot be quantized on the platform. 
- The program for inference is available 





<br><br>


# Benchmarking of PINTO Models on STM32 MCU with Neural-ART™



| Model Name                | Extention | MACC | Flash Size (Total) | RAM Size (Total) | Inference Time | Input Size | Landmarks | Inference Program | Additional Notes         
|---------------------------|-----------|------|--------------------|------------------|----------------|------------|-----------|-------------------|--------------------------
|version-RFB-320-int8       | onnx      |   101,269,024  |      ?             |    ?             |        ?       |   240x320  |  &#10008; |   &#10004;        | NOT IMPLEMENTED: Onnx exporting model with quantized unsigned integer format is not supported
|version-RFB-320            | onnx      |   ?  |      ?             |     ?            |        ?       |   240x320  |  &#10008; |   &#10004;        | NOT IMPLEMENTED:Order of dimensions of input cannot be interpreted
|version-RFB-640            | onnx      |   ?  |      ?             |     ?            |        ?       |    480x640 |  &#10008; |   &#10004;        | NOT IMPLEMENTED:Order of dimensions of input cannot be interpreted
| /DSFD-VGG-Lightnet-0.5/     | tflite    |   ?  |   665 KB         | 5.2 MB           |       ?        | 320x320    | &#10008;  |   &#10004;        |INTERNAL ERROR: Unknown dimensions: CH 
| /DSFD-VGG-Lightnet-0.75/    | tflite    |   ?  |   914 KB         | 5.2 MB           | ?              | 320x320    |  &#10008; |   &#10004;        | INTERNAL ERROR: Unknown dimensions: CH for inferece
| dsfd_256x256_float16_quant| tflite    |   ?  |     ?              |     ?            |    ?           |  256x256   |  &#10008; |   &#10008;        | INTERNAL ERROR: 'FLOAT16'
| dsfd_256x256_weight_quant | tflite    |   ?  |     ?              |      ?           |     ?          | 256x256    |  &#10008; |   &#10008;        |  NOT IMPLEMENTED: Unsupported layer types
| dbface_480x640_weight_quant| tflite   |   ?  |     ?              |     ?            |      ?         |   480x640  |  &#10004; |  &#10004;         |  INTERNAL ERROR: 'NoneType' object has no attribute 'shape'
| dbface_512x512_weight_quant| tflite   |   ?  |     ?              |    ?             |      ?         |   512x512  |  &#10004; |   &#10004;        |  INTERNAL ERROR: 'NoneType' object has no attribute 'shape'
| dbface_keras_480x640_weight_quant_nhwc| tflite|2,136,463,975| 7.1 MB          |    30.8 MB       |     5.28e+4 ms|  480x640   |  &#10004; |   &#10004;        |   
| dbface_keras_480x640_integer_quant_nhwc| tflite|2,086,146,121| 3.1 MB         |  13.1 MB         |     486.2 ms   | 480x640    | &#10004;  | &#10004;          |
| dbface_keras_256x256_weight_quant_nhwc | tflite|455,834,343| 7.1 MB         |  7.4 MB          |   1.212e+4 ms  | 256x256    | &#10004;  |  &#10004;         |
| <span style="color: green;">dbface_keras_256x256_integer_quant_nhwc</span>| tflite|445,098,409| 3.57 MB        |     2.14MB       |   27.72 ms     |  256x256    | &#10004;  |  &#10004;         | Almost high temporal jitter
| /Face_Landmark/yolov4_headdetection_480x640|onnx|54,597,774,206|               |    ?             |    ?           |   480x640  |  &#10008; |  &#10004;         | INTERNAL ERROR: node_1113 of type Tile has not parameter repeats. Available parameters are dict_keys(['output_shape_values']). It did not worked even with random quatization of the software.
| face_landmark_Nx3x160x160| onnx        |?|    ?          |    ?             |      ?         |   160x160  |   &#10004;|   &#10004;         |  This is a model for predicting dense 2d face mesh that should be coupled with a face detector. NOT IMPLEMENTED: Order of dimensions of input cannot be interpreted.
|/Face_Landmark/model_integer_quant|tflite |43,910,554| 1.1 MB        |  677 KB          |        39.72 ms        |   160x160  |   &#10004; |    &#10008;      | Same as previous model but different format
| <span style="color: green;">centerface_Nx3xHxW</span>| onnx              | 2,377,640  |  7 MB         |   18 KB          |        31.63 ms        |   Any      |   &#10004; |    &#10004;      |
|centerface_Nx3xHxW_PerChannel_quant_random_1| onnx |  2,377,640 | 2.2 MB               |       72 KB             |  10.2 ms   |      Any      |   &#10004; |    &#10004;    |  
| scrfd_500m_480x640| onnx              | 510,249,146    |  ? |       ?         |         ?              |  480x640   |   &#10008; |  &#10004;        | Error
|scrfd_500m_bnkps_480x640| onnx         | 562,123,406    |  ? |       ?         |         ?              |   480x640  |   &#10004; |     &#10004;     | Error 
| /SCRFD/model_float32| tflite         | 561,568,106       |  2.6 MB| 19.8 MB     |     1.564e+4 ms        |   460x640  |   &#10004; |   &#10004;       |  
| scrfd_500m_240x320  | onnx           | 130,080,266       |  ?     |  ?          |     ?                  | 240x320    |    &#10008;|   &#10004;       | Error
| /YuNet/model_float16_quant| tflite| ?                    |  ?     |   ?         |    ?                   | 120x160    |   &#10004; |  &#10004;        | INTERNAL ERROR: 'FLOAT16'
| /YuNet/model_weight_quant|  tflite| 25,053,904           |  434KB |  971KB      |        770 ms              |  120x160   |  &#10004;  |   &#10004;   |
| face_detection_yunet_120x160| onnx| 25,092,926           |  ?     |    ?        |   ?                        | 120x160    |   &#10004; |   &#10004;   |INTERNAL ERROR: arr must be of type np.generic or np.ndarray, got <class 'list'>
|yolov7_tiny_head_0.768_post_480x640| onnx|         ?      |  ?     |     ?       |           ?                |   480x640  |  &#10008;  |  &#10004;    |  NOT IMPLEMENTED: Unsupported layer types: NonMaxSuppression
| yolov7_tiny_head_0.768_480x640|     onnx| 4,953,211,746  |  23 MB |    21 MB    |              ?              |   480x640  |  &#10008;  |  &#10004;   |   Build Error 
| yolov7_tiny_head_0.768_480x640_PERCHANNEL_QUANT_RANDOM | onnx| 4,953,211,746 | 6.2 MB |       8.3MB   |    731.8 ms    |  480x640  |  &#10008;  |  &#10004;   |
|directmhp_cmu_s_640x640|        onnx                    |             ?       |    ?   |         ?     |       ?        |    640x640|  &#10008;  |  &#10008;   | NOT IMPLEMENTED: Unsupported layer types: ScatterND. This is the network with no post processing inside the model. It outputs the pose of the head. The models with post processing are not applicable on the target(Layers Not Supported).
|directmhp_agora_s_640x640|  onnx                        |            ?        |    ?   |         ?     |        ?       |  640x640  | &#10008;   | &#10008;    | NOT IMPLEMENTED: Unsupported layer types: ScatterND
|retinaface_mobilenet0.25_Final_1x3x256x256_fixed|  onnx |        162,949,232  |  1.7 MB | 1.5 MB       |       5203 ms         |  256x256  | &#10004;   |  &#10008;   | There are a lot of variations of this model family.
| retinaface_mobilenet0.25_Final_1x3x256x256_fixed_PERCHANEEL_QUANT_RANDOM| onnx|   162,949,232 | 737 KB | 620 KB |       ?      |  256x256  | &#10004;   |  &#10008;   | Error for inference on the platform. 
| faceboxes_1x3x192x320 |                               onnx |  56,643,175      |  3.9 MB       | 1.8 MB |         ?              |  192x320  |  &#10008;  |  &#10008;   | Build Error. There are several variations of this model family based on input size. 
| faceboxes_1x3x192x320_PerChannel_quant_random_1 |    onnx  | 56,643,175       |  1.1 MB       | 913 KB  |          7.537 ms             |   192x320 | &#10008;  | &#10008;   | The original model did not work for inference but the random quantized one did.
| mobilefacenet_Nx3x112x112 |                       onnx  |   226,881,224       | 4 MB          | 4 MB    |              5337 ms                 | 112x112    | &#10004; | &#10004;  | This is a network for dense face mesh (Landmarks). It needs to be coupled with a face detector. Error for quantizing on the platform. 


- The models are evaluated on the target device on the STM32 edge AI platform withou using the random quantization option, keeping the original precision of the model file provided
- Except some of them that are appended by "PERCHANNEL_QUANT_RANDOM"

<br><br>

# ONNX model zoo

- There are models available for Emotion recognition, Age, and gender classification 

<br><br>

# B-FPGM ( Latest face detector, 2025)

- Developed a code to load the models saved in `.pth` format for inference. 
- The work is about prunning a two state-of-art models, however, the resulting models seem not robust during inference and also seem slow. 
- The input size of the model is still vague for me.
- Run this code for inference > `/B-FPGM/EResFD-main/demo.py`
- [Link to the github repo](https://github.com/IDT-ITI/B-FPGM/tree/main)

<br><br>

# FeatherFace ( Latest face detector, 2025)

- [Link to github repo](https://github.com/dohun-mat/FeatherFace?tab=readme-ov-file)
- No, pre-trained model is provided. 

<br><br>

# Comparative Analysis of CenterFace and BlazeFace for Face Detection Applications  

- ***[Check out this repository](https://github.com/Maaz77/face-detector-quantization/tree/main?tab=readme-ov-file)***

## WIDER FACE Dataset Evaluation  
The WIDER FACE benchmark, comprising images stratified into Easy, Medium, and Hard subsets based on detection difficulty, provides critical insights into model robustness:  

| Model       | Easy Set (AP) | Medium Set (AP) | Hard Set (AP) |  
|-------------|---------------|-----------------|---------------|  
| CenterFace  | 93.5%         | 92.4%           | 87.5%         |  
| BlazeFace   | 89.6%         | 85.3%           | 72.8%         |  

CenterFace demonstrates superior performance across all difficulty tiers, with a 15.6% relative improvement on the Hard set containing occluded and low-resolution faces. The anchor-free design mitigates scale sensitivity issues prevalent in anchor-based methods, allowing consistent detection across diverse aspect ratios. BlazeFace, while maintaining competitive accuracy on Easy/Medium sets, shows decreased robustness for small faces due to its fixed anchor configurations and spatial resolution constraints.  


## Operational Efficiency Tradeoffs  

While accuracy metrics favor CenterFace, implementation context dictates model suitability. CenterFace operates with a 7.2MB model size and achieves 30 FPS on an Intel i7-6700 CPU, whereas BlazeFace’s 1.5MB architecture enables 200+ FPS on mobile GPUs like the Adreno 640. BlazeFace’s depthwise convolution architecture reduces multiply-accumulate (MAC) operations by 4.8× compared to CenterFace’s standard convolutions, optimizing power efficiency for battery-dependent devices. However, CenterFace natively supports facial landmark detection (5 points) with minimal accuracy degradation, while BlazeFace requires auxiliary models for extended facial analysis.



## Failure Mode Analysis  

### CenterFace Limitations  
CenterFace’s heatmap refinement process introduces 12–15ms latency per image on CPU platforms, constraining real-time applications. The joint optimization of centerness prediction and bounding box regression losses also increases training complexity, requiring careful hyperparameter tuning to ensure stable convergence.  

### BlazeFace Limitations  
BlazeFace struggles with scale sensitivity, as its fixed anchor scales cannot adapt to extreme zoom variations beyond a 10:1 scale ratio. This reduces Hard set performance by approximately 23% under >40% facial occlusion, as the model lacks explicit occlusion-handling mechanisms.  

## Conclusion  

For applications requiring high accuracy in challenging conditions (surveillance, crowd analysis), CenterFace is superior with its anchor-free design and multi-task learning. BlazeFace is better suited for mobile applications needing real-time processing despite accuracy compromises in complex scenes. Choose CenterFace for high-density environments or low-resolution footage analysis, and BlazeFace for smartphone/AR applications where low latency and power efficiency are critical.

<br><br>

----
----
----

# Head Pose Estimation

## [Real Time Head Pose Estimation](https://github.com/yakhyo/head-pose-estimation/tree/main?tab=readme-ov-file)
- This repo contains several models for head pose estimation from heavy to light weight models. Based on ResNet and MobileNet.
- It must be coupled with a face/head detector
- Tried to load the pytorch MobileNetV2 version and convert it to `.onnx` network to deploy it on the STM32N6 device but i got and error regarding the output dimension. 
- I could not figure out why there was the problem 
- Tried to use the `onnx2tf` package to convert the onnx model to `.tflite` and then deploy it on the device, however, the same error showed up with the tflite model on the ST platform.

## DMHead
- Apparently, the models in this PINTO model zoo repo are based on MobileNetV2 but only the `dmhead` versions.
- Tried to deploy the models on the platform, however, there were an error. 
- Used `onnx2tf` to deploy the tflite version of these models on the platform, again there was an error.
- Check out the `Models Found` directory.

```diff
- My intention was in this step to find the lightest version of a head pose estimator model to couple it with BlazeFace or CenterFace to assess the performance and latency.
- I was targeting the models based on MobileNet like the two aforementioned steps. But, yet did not found any model that is compatible with ST platform.
```

## [Head Pose Estimation by landmarks](https://github.com/yinguobing/head-pose-estimation/tree/master)
- In this work a network is used to extract 68 landmarks from the region of the face.
- Another rule based scritps is used to output the head pose having these face landmarks.
- The landmark model is tested on the Platform and it gives 25ms latency. The model seems lights weight. 
- ***I have to test is for inference then if it is good i will use it for quantization***

- `LwPosr – Lightweight Efficient Fine-Grained Head Pose Estimation` and `MobileNet` are the most top efficient networks for pose estimation.


# Head Pose Estimation Datasets 

## [Biwi Kinect Head Pose Database](https://www.kaggle.com/datasets/kmader/biwi-kinect-head-pose-database/data)
  - The dataset is a bit complicated.
  - The images are not only the region of face. They are captures from a position of 1 meter from the subject. 
  - In this [huggingface link](https://huggingface.co/datasets/ETHZurich/biwi_kinect_head_pose) there is a code provided to load the dataset.
  - In this [google colab notebook](https://colab.research.google.com/drive/14AWKhaC7H5pG7I53D-NuknfwCdEJYMR1?usp=sharing) I have tried to load the dataset.
  - But, since the dataset is annotation is a bit complicated and the images are not image of the face, I skip it for now.
  - The BIWI Kinect Head Pose dataset does not explicitly provide ground truth face bounding boxes directly in the dataset files.
  - Head poses are estimated by in-depth information from Kinect.(DirectMHP paper)
  - A major drawback of these datasets is that the Euler angles are narrowed in the range of (−99◦,99◦). (DirectMHP paper)


## [300W-LP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
  - The synthesized large-pose face images from 300W
  - The dataset contains some parameters from the camera, shape, and so on. 
  - The pose of the head is provided by radian angles.
  - The dataset is also accessible by [Tensorflow](https://www.tensorflow.org/datasets/catalog/the300w_lp).
  - The images of the face are derived from rotating a 3D face mesh of face to capture large poses so that in these cases the image is deformed. 
  - [Link to google colab](https://colab.research.google.com/drive/1IsjYZ-IIWumHV7YThW_bY6ibrNcI7kIR?usp=sharing)
  - The Basel Face Model is a 3D Morphable Model (3DMM) — a statistical model of 3D human face shape and texture.

## [ETH-XGaze](https://ait.ethz.ch/xgaze)

  - It is a very good dataset for gaze estimation. 
  - However, the label for gaze direction is provided in 2d (Pitach and Yaw), the roll is not provided.
  - They normalize the face region so that the two eyes are horizontaly aligned so the roll is cancled out.
  - Since we want the head pose estimation and face detection be unified in a single network, the dataset is not suitable for out task for now. 
  - [Github Link](https://github.com/xucong-zhang/ETH-XGaze)

## [DD-Pose](https://dd-pose-dataset.tudelft.nl/eval/overview/statistics)

  - Images of human driving, very good dataset
  - Labels are 6 DoF. 3 for rotation (YPR), and 3 for translation: how the head is positioned in 3D space along X, Y, and Z axes.
  - The 3D translation vector (t_x, t_y, t_z) tells you the position of the head’s center (usually the nose bridge or eye center) relative to the camera.
  - We should request to get access to the dataset

## UET-Headpose

  - Seems a good dataset but the download link is not provided. 

## [AFLW2000](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)

  - This dataset does not contain explicitly the head pose angles, however, it contains 68 3d points landmarks of the the face. 
  - It is possible to use these 3d points to extract the head pose by solving `A Perspective-n-Point (PnP)` problem (or a fitting model) and also having the intrinsic camera parameters. 
  - It is also [available in tensorflow](https://www.tensorflow.org/datasets/catalog/aflw2k3d)

## AGORA & CMU Dataset

  - These are datasets of video of som people carrying on an activity.
  - In the paper `DirectMHP`, these dataset is to build up another version of the datasets so that they contains the human faces plut head pose labels.
  - For now, we can skip these datasets. 
  - [Github Link](https://github.com/hnuzhy/DirectMHP?tab=readme-ov-file#mphpe-datasets)


## [Good to go Datasets](https://github.com/shamangary/FSA-Net/blob/master/README.md#1-data-pre-processing)

  - The images spatial size are `64x64`
  - Only images of the faces regions 
  - The biwi dataset has a subset namd `BIWI_NoTrack.npz`. This dataset, contains the same images as in the other two images but excluding the frames that the face detector failed to detect the faces. 


# Experiment on Benchmarking Head Pose Estimator SOTA Models from PINTO Model Zoo

## WHENet 
  - The .h5 model from the [official Repository](https://github.com/Ascend-Research/HeadPoseEstimation-WHENet) of the work is used for Benchmarking.
  - Model uses classification instead of regression - outputs probability distributions over discrete angle bins
  - Yaw head: 120-dimensional output (likely covering -180° to +180° range with 3° per bin)
  - Pitch and Roll heads: Each 66-dimensional output (covering smaller angle ranges since head has more limited pitch/roll movement than yaw)
  - the results from both the PINTO Model and the official repository is available. 
  - did not double check the coding.

## Head Pose Estimation Adas-0001
  - A vanilla CNN network for regressing the angles of the head pose.
  - The three version of the model (onnx, tflite, openvion) are evaluated on the BIWI test set. 

## 6DRepNet360 
  - the onnx version of the model obtained from PINTO MODEL ZOO is evaluated on the BIWI test set.
  - ***It is important to mention that, the PINTO model outputs the 3 scalars related to angels while the official models output 3*3 rotation matrix.**
  - [In the official Repo](https://github.com/thohemp/6DRepNet360/tree/master) there are some codes to read the benchmark head pose datasets and their labels.
  - As far as i can see, the code i developed for evaluating the models from the official repo is correct, considering the rotation matricies and angles of the labels.  
  - The code i used for evaluating the models from the official repo is a bit complicated. 
  - Checked the order of matching labels and the output of the model. the one in the code is correct

## 6DRepNet 
  - [Official Repo](https://github.com/thohemp/6DRepNet)
  - there are pytorch trained model files in the repo
  - there are links to head pose datasets in the repo 
  - ***skipping it as of now because this work is a older version of the 6DRepNet360 model.**


## DMHead
  - the evaluation results are available. 
  - the models used are from PINTO.
  - the [original repo](https://github.com/PINTO0309/DMHead?tab=readme-ov-file) is also a PINTO repo

## LightWeight-HeadPose-Estmiation

  - regarding the models on PINTO MODEL ZOO, i am not sure about the order of output angles.
  - [Official Repo](https://github.com/Shaw-git/Lightweight-Head-Pose-Estimation/tree/main)

## Opal23 HeadPose 

  - The model on the PINTO Model zoo is evaluated, however the results are so pessimistic.
  - Tried a lot to run and evaluate the model on the official repo as it is mentioned in the README of the official repo, however, it was lacking some model file dependencies for SSD framework so it was not working.
  - Tried to alter the code to use only head pose detection network on the test dataset but it was not working. The framework is exteremly complicated.

## DirectMHP 

  - The thing about this model is that the model do face detection as well, so in most of the cases in the test images it could not detect anyface so the outputs for the pose angles are useless. To this end, I exclude this model for now from benchmarking.
  - The code to evaluate the model on the test set for models obtained from PINTO MODEL ZOO is available. The models are not performing well on the testset. 

# Different protocols/paths for training and evaluating the head-pose regressor model 

## Protocol 1
  - Train on `300w-lp`:
    - A. Test on `AFLW2000` 
    - B. Test on `BIWI_Test` (30% of `BIWI`)
## Protocol 2 : 
  - Train on `BIWI_Train`
     - A. Test on `BIWI_Test`
     - B. Test on `AFLW200` -> This approach is not practiced in the SOTA
## Protocol 3 (not practiced in the SOTA): 
  - Train on `BIWI_Train` + `300w-lp` 
    - A. Test on `BIWI_Test`
    - B. Test on `AFLW200`

```diff 
+ So far, the protocol 2.A is done. Now, we are aiming to do 2.B evaluation to see if the performance is satisfying, otherwise we go to protocol 3. 
```

## Protocol 2.A Benchmarking SOTA models

<!-- | Method   | Yaw  | Pitch | Roll | MAE  |
|----------|------|-------|------|------|
| HopeNet  | 3.29 | 3.39  | 3.00 | 3.23 |
| FSA-Net  | 2.89 | 4.29  | 3.60 | 3.60 |
| TriNet   | 2.93 | 3.04  | 2.44 | 2.80 |
| FDN      | 3.00 | 3.98  | 2.88 | 3.29 |
| 6DRepNet | 2.69 | 2.92  | 2.36 | 2.66 |
| DeepHeadPose| 5.67 | 5.18 | - | -     |
| SSR-Net-MD| 4.24 | 4.35 | 4.19 | 4.26 |
| VGG16|    3.91   | 4.03 | 3.03 | 3.66 | 
|FSA-Caps-Fusion| 2.89 | 4.29 | 3.60 | 3.60|
|HeadPosr EH64| 2.59 | 4.03 |3.53 | 3.38 |
|THESL-Net |	2.53| 	3.08|	2.95	|2.85| 
|MFDNet |2.99	|3.68|	2.99|	3.22|
|ST-ViT | 3.27 | 2.82| 3.12 | 3.07|  -->


| Method | Yaw | Pitch | Roll | MAE |
|--------|-----|-------|------|-----|
| 6DRepNet | 2.69 | 2.92 | 2.36 | 2.66 |
| TriNet | 2.93 | 3.04 | 2.44 | 2.80 |
| THESL-Net | 2.53 | 3.08 | 2.95 | 2.85 |
| ST-ViT | 3.27 | 2.82 | 3.12 | 3.07 |
| MFDNet | 2.99 | 3.68 | 2.99 | 3.22 |
| HopeNet | 3.29 | 3.39 | 3.00 | 3.23 |
| FDN | 3.00 | 3.98 | 2.88 | 3.29 |
| HeadPosr EH64 | 2.59 | 4.03 | 3.53 | 3.38 |
| FSA-Net | 2.89 | 4.29 | 3.60 | 3.60 |
| FSA-Caps-Fusion | 2.89 | 4.29 | 3.60 | 3.60 |
| VGG16 | 3.91 | 4.03 | 3.03 | 3.66 |
| SSR-Net-MD | 4.24 | 4.35 | 4.19 | 4.26 |
| DeepHeadPose | 5.67 | 5.18 | - | - |




- References: 
  * https://github.com/thohemp/6DRepNet?tab=readme-ov-file
  * [FSA-Net: Learning Fine-Grained Structure Aggregation for Head Pose Estimation from a Single Image](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_FSA-Net_Learning_Fine-Grained_Structure_Aggregation_for_Head_Pose_Estimation_From_CVPR_2019_paper.pdf)
  * [HeadPosr: End-to-end Trainable Head Pose Estimation usingTransformer Encoders](https://arxiv.org/pdf/2202.03548#:~:text=RGB,38)
  * [An Improved Tiered Head Pose Estimation Network with Self-Adjust Loss Function](https://www.mdpi.com/1099-4300/24/7/974#:~:text=FSA,Net%202.53%203.08%202.95%202.85)
  * [Towards Human-CenteredAutomated Driving: A Novel Spatial-Temporal Vision Transformer-Enabled Head Tracker](https://dspace.lib.cranfield.ac.uk/server/api/core/bitstreams/bf71427b-5a11-4d18-bdb1-ab5d4378a457/content)
- There is other method  [Self-Attention Mechanism-Based Head Pose Estimation Network with Fusion of Point Cloud and Image Features](https://pmc.ncbi.nlm.nih.gov/articles/PMC10747419/#:~:text=that%20the%20average%20absolute%20errors,BIWI%20head%20pose%20recognition%20accuracy) that is performing much better than the above methods.


## Protocol 2 Benchmarking our models for `96dim` feature 

| Model ID | MAE (on `BIWI_Test`)      | # Params | MAE on `AFLW2000` | Dataset Train
|----------|---------------------------|----------|-------------------|--------------------------|
| rd93oeou | 2.32                      | 101155   | 8.09              |BIWI_Train + BIWI_NoTrack | 
| wnfcrqss | 2.34                      | 80019    | 8.38              | BIWI_Train + BIWI_NoTrack| 
| za1bxuzn | 2.54                      | 34707    | 8.12              | BIWI_Train + BIWI_NoTrack| 
| cl4obelj | 3.68                      | 3683     |  8.67             | BIWI_Train
|yav3m4y3  | 3.74                      | 3683     | 8.35              | BIWI_Train




