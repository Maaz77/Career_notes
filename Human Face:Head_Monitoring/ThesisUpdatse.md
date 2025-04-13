# Problem Definition 

## Human Face Monitoring from infrared input stream using a pipeline of neural network models to be deployed on `STM32N6` MCU. 

- To be more clear, we are aiming for a system to be run on `STM32N6`, which is a product of the `STMicroelectronics` company, that is able to detect a human face from an input stream of video to perform the following subtasks: 
    - Head Pose Estimation.
    - Gaze Tracking.
    - Detect if the person is talking on the phone.
    - Detect if the person is eating/drinking.
    - Detect if the person is smoking.
    - Detect if the person is yawning or not.
    - and so on.

- We are currently focusing more on the first two tasks.
- A demo of the supposed system developed by the `Qualcomm` company can be found [in this link](https://www.youtube.com/watch?v=-zu_uLDMV90).
- A substantial use case of such a system could be monitoring the driver of a vehicle for safety purposes.

# System Requirements 

- The system should perform with high relative accuracy in real time. 
- The system should have a small memory footprint and low power consumption. 
- From a practical point of view, the neural network models must be in INT8 quantized precision (Weights and Activations) to be compatible with the device requirements. 

# Roadmap to Build the System 

1. Identify a face detector model:

    - The model should be small in terms of the number of parameters and activation memory requirements.
    - The model should be robust with high accuracy, performing in real-time inference. 
    - The challenge here is to find an equilibrium point between complexity and efficiency. 
    - The performance of the models should not drop significantly after INT8 quantization.

2. Identify a dense face mesh predictor model (2D or 3D): 

    - We need a model that takes an image of a face region and outputs a dense face mesh of the face. 
    - The model should be able to locate the center of the iris for the task of gaze tracking. 
    - The model should be robust with high accuracy while requiring the least amount of memory. 
    - The model's performance should not be compromised after INT8 quantization. 

3. Perform the final subtasks given the dense face mesh of a face: 

    - This step could be done using neural network models or rule-based programs. 

# Updates on My Activities 

## Mid-April 2025 

1. At the beginning, I conducted a literature review study in the field of Object Detection to:
    - Have a solid understanding of how Object Detection models work or are trained.
    - Identify the SOTA models and investigate their properties in terms of accuracy and complexity. 
    - Conclude a list of selected models suitable for our task with their features. 

2. Conducted a shallow study in the field of face mesh detector models to gain an understanding of how the models work or are trained, specifically the works provided by Google.

3. Investigated and studied the frameworks and GitHub repositories that provide the tools for tasks related to face detection and face mesh detection. 
    - MediaPipe from Google. 
    - PINTO Model ZOO.
    - Dlib.
    - InsightFace.
    - And so on. 

4. Investigated all the models for the tasks of Face Detection, Head Pose Estimation, and Dense Face Mesh Prediction provided by [PINTO MODEL ZOO](https://github.com/PINTO0309/PINTO_model_zoo).
    - The main goal at this step was to find face detector model files, but sometimes I came across models for the other two tasks as well. 
    - To this end, around `40 pre-trained models` of SOTA were analyzed. 
    - The models are provided in different formats and frameworks. 
    - The inference code for each model was developed partially or, in some cases, from scratch to have a demo of each model.
    - An important subtask at this step was to test if the model files (`.h5`, `.tflite`, `.onnx`, etc.) are compatible with the `STMCubeAI` platform. This platform is developed by the company for optimizing and deploying neural network models on the ST MCUs. 
    - The result of this step was not only having a demo for each model but also a long table of benchmarking of the identified models containing information about model complexity, memory requirements on the device, flash requirements on the device, and so on. 
    - Finally, we came up with a few selected models: `BlazeFace` (from Google and almost the smallest), `CenterFace`, and `dbface`. 

5. Quantized the few selected models yielded in the previous step into INT8 bit precision. 

    - At this step, the `onnx2tf` tool provided by PINTO was utilized to quantize the models into 8-bit precision using a custom calibration dataset. 
    - For creating the calibration dataset, I captured images of myself from the webcam camera of my laptop.
    - The custom calibration dataset was used to quantize the `CenterFace` and `dbface` models but not the `BlazeFace` model.
    - The `BlazeFace` model was quantized using a different dataset found on the internet (not a benchmark dataset).
    - [The GitHub link](https://github.com/Maaz77/face-detector-quantization)
----
---
***End of the First Stage of the Project***

----
----

6. Due to some experiments on face mesh models quantized (MediaPipe Face Mesh Model), we realized that the drop in accuracy after quantization is high, so we decided to change our approach. To be more specific, we endured some drop in accuracy from the face detector model due to quantization, and again we experienced a significant drop in performance from the face mesh detector due to quantization. Therefore, as of now, we decided to shift to another approach, which is having a unified network to perform at least one or two of the subtasks instead of pipelining two or three quantized neural networks. In the latter case, we guess that the performance of the system would not be acceptable. 

7. ***We came up with an idea of using `BlazeFace`'s latest feature maps, before the regressor and classification heads, to train a model for head pose estimation, and finally, join this network to the BlazeFace detector to have a unified model for face detection and head pose estimation tasks. (Transfer Learning)***

8. Performed a study on the SOTA models for head pose estimation. I aimed to:
    - Find the smallest head pose estimation models.
    - Find the works that proposed unified models for both face detection and head pose estimation.
    - As of now, I could only find one work that proposed a unified network for the tasks (`DirectMHP`).

9. Identified head pose estimation benchmarking datasets for experimenting with the idea to see if the features extracted by the BlazeFace model are informative for the task of head pose estimation. 

11. Identified the protocols or conventions practiced by the authors in the field for using specific datasets for training the head pose regressor model and specific datasets for evaluation. 

12. Extracted feature maps from the BlazeFace model from each dataset (images of face regions and head pose labels) to build datasets of feature maps and head pose labels for training a head pose regressor model.

13. Conducted many experiments for hyperparameter tuning and model architecture design. 

14. ***Following a famous protocol for training and evaluating a head pose regressor model, our trained model is outperforming all the SOTA models with a little tweak :)***

### To-Do List: 

1. The BlazeFace model has two prediction heads. So far, only the feature maps from one head have been extracted, and the corresponding model has been trained and evaluated. We need to extract feature maps from the other head as well to train another regressor model. 
2. Follow other different protocols for benchmarking our model (training + evaluation).
3. Create a demo of the head pose estimator model.
4. Join the two regressor models to the `BlazeFace` model to have a unified network for face detection and head pose estimation. 
