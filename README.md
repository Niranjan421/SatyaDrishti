<a id="readme-top"></a>

# SATYADRISHTI

[NOTE: The trained model and dataset has not been uploaded due to GitHub's file size restriction, which limits uploads to files smaller than 25MB.(ONLY FOR FACE SWAP DEEPFAKE DETECTION)] 

## Explanatory Video for SatyaDrishti
https://github.com/user-attachments/assets/285b7190-0cd2-41c7-86f0-3acd8b8515f8



## Index: 
- [About The Project](#About-The-Project)
- [Installation](#Installation)

<!-- ABOUT THE PROJECT -->
## About The Project
Certainly! Here’s a concise breakdown of a deepfake audio detection project using an RNN model in four short points:

### Data Collection and Preprocessing:

The Deepfake Detection System begins with collecting a comprehensive dataset of both real and manipulated audio and video samples. For video, each file is broken down frame-by-frame to extract facial features, expressions, and transitions. For audio, advanced signal processing techniques are used to convert speech into measurable features such as MFCCs, pitch, tone, and modulation. These extracted features serve as structured inputs for deep learning models, enabling them to identify subtle inconsistencies that differentiate authentic media from synthetically generated ones.

### Model Development:

The system employs separate models for video and audio detection. For video, convolutional neural networks (CNNs) and frame-wise analysis detect unnatural expressions, lip-sync mismatches, and visual tampering. In parallel, recurrent neural networks (such as LSTMs or GRUs) analyze sequential audio features to identify anomalies in speech patterns, rhythm, and acoustic signatures that often indicate a voice deepfake. Both models are trained to classify media as real or fake, and are capable of operating in real time to provide immediate feedback.

### Model Evaluation:

The performance of the detection models is rigorously assessed using industry-standard evaluation metrics, including accuracy, precision, recall, and F1-score. For videos, the system provides visual outputs highlighting suspicious regions, while for audio, it returns an authenticity score based on detected anomalies. Continuous benchmarking ensures the models remain robust against the latest advancements in deepfake generation techniques.

### Deployment and Monitoring:

The trained system is deployed as an interactive platform where users can upload or record live audio and video for deepfake analysis. The interface is designed for ease of use by journalists, enterprises, and the general public. Behind the scenes, the models continue to learn and adapt by training on newly encountered deepfake data, ensuring up-to-date performance. Real-time processing and adaptive learning make the system reliable and scalable for long-term use in combating digital misinformation.



# Installation
To install it, download all the files in a ZIP format, check the image location according to your device, and then click on the app.py file to run it,your site will be up and running.


<p align="center">(<a href="#readme-top">back to top</a>)</p>
