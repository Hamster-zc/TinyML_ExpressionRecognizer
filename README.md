# TinyML:EfficientNet-based Facial Expression Recognition Model
It is a repository for our grade-one-program
## Project Description
This project focuses on building an automatic facial expression recognition model using EfficientNet and successfully deploying it on ESP32. It aims to achieve efficient and accurate facial expression recognition suitable for resource-constrained embedded environments.
## Key Features
EfficientNet Model Utilization: Leverage the architecture benefits of EfficientNet for efficient feature extraction and recognition of facial expression images.
ESP32 Deployment Optimization: Optimize the model according to the hardware characteristics of ESP32 to ensure stable operation on resource-limited devices.
Multi-expression Classification Support: Cover multiple common expression classifications to meet the needs of facial expression analysis in different scenarios.
## Dataset Information
Data Sources: Use publicly available facial expression datasets FER-2013, which contain a large number of labeled facial images across various expression categories including anger, disgust, fear, happiness, sadness, surprise, and neutrality.
Data Preprocessing: Standardize images to fit EfficientNet input requirements and augment data through operations like rotation and flipping to enhance model generalization.
## Model Training and Evaluation
Training Framework: Train the model using deep learning frameworks like TensorFlow with GPU acceleration to improve efficiency.
Evaluation Metrics: Comprehensively assess model performance using metrics such as accuracy, recall, and F1 score to ensure good performance in facial expression recognition tasks.
## Deployment Process
Model Conversion: Convert the trained EfficientNet model to a format suitable for ESP32 deployment, such as TensorFlow Lite.
Code Adaptation: Write inference code compatible with ESP32 to implement functions like model loading, image preprocessing, and prediction, ensuring correct model operation on the device.
## Getting Started
Environment Setup: Set up the development environment as required, including installing necessary libraries and tools.
Model Loading: Load the converted model file into ESP32 for expression recognition preparation.
Image Input: Acquire facial images via camera or other means and preprocess them.
Inference and Prediction: Use the loaded model to recognize expressions in input images and obtain prediction results.
## Notes
Hardware Compatibility: Ensure the ESP32 device used is compatible with the project code and model. Some devices may require additional drivers or configurations.
Performance Optimization: Depending on the actual application scenario, further optimize the model through quantization, pruning, etc., to improve runtime efficiency on ESP32.
Data Security: When handling facial image data, comply with relevant privacy regulations to ensure legal use and storage of data.
## Project Contributions
Contributions from developers are welcome, including but not limited to code optimization, feature expansion, and documentation improvement. Before contributing, please read the project's contribution guidelines to understand relevant norms and processes.
