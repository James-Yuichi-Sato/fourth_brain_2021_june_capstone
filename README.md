# Smart Annotation Pipeline for Video Sequences

## Fourth Brain 2021 June Capstone

## Project Description

In this project, our aim is to improve the video annotation workflow by removing redundant frames in video, as well as removing noisy frames and gaining insights into the videos using ML model detection as well as the metrics used to remove noisy and duplicate frames.

We developed the workflow using the dashcam videos openly available in the [JAAD Dataset](https://github.com/ykotseruba/JAAD).

This workflow is intended to provide insight to improve the data extraction - data validation - data preparation stages in an MLOps workflow.

<img src="media\mlops-continuous-delivery-and-automation-pipelines-in-machine-learning-3-ml-automation-ct.png" alt="media\mlops-continuous-delivery-and-automation-pipelines-in-machine-learning-3-ml-automation-ct.png"></img>

https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning



## Components of Workflow

### Overview

<img src="media\Workflow.drawio.png" alt="media\Workflow.drawio.png"></img>







### Data Storage

This workflow uses the GCP Cloud Storage library to bulk store the video footage for ingress, and exports the separated frames and detection outputs to a subfolder named by video. The data is stored on the local machine during runtime.



Before:

![GCP Before](media\GCP_before.png)



After:

![GCP After](media\GCP_after.png)

### Frame Separation

Software pulls frames from mp4 videos using ffmpeg. The frames are stored in a folder with the same name as the video.

### Noisy Frame Reduction

The software calculates the Laplacian variance of all of the frames of the video using OpenCV, then removes frames with 5% more Laplacian variance than the median. This value was chosen as it would prevent video with multiple types of scenes removing frames due to a scene change rather than noisiness. This generally removes around 

### Similar Frame Reduction

The software uses the structural similarity function from OpenCV to compare frames. It first calculates the similarity between each frame sequentially first, than delete frames that are more than 95% or 105% of the median similarity similar, whichever is greater.

### Metadata Tagging for Storage and Retrieval

The software uses the open-source [Faster RCNN Resnet50 model trained on Coco dataset](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz) to classify objects in each frame. The detection classes are saved in a text file with the same name as the frame it was detecting in the frames folder.

<img src="media\GCP_frames.png" alt="media\GCP_frames.png">

Frames with detection metadata after workflow is completed

## Running the Software

### Weights and Biases

This software runs 

<img src="media\weights_and_biases3.png" alt="media\weights_and_biases3.png">

### GCP IAM Key

The software uses a `gcp_key.json` file to access the google cloud storage instance. You can create the file on [Google Cloud](https://cloud.google.com/iam/docs/creating-managing-service-account-keys).

### FastAPI

This software exposes FastAPI on port 8000 of the machine. You can access all of the functions using `http://<EXTERNAL_URL>:8000/doc/`. You can frame clean individual videos in the GCP bucket using the `clean_single_video` call. You can frame clean all videos in the GCP bucket using the `clean_bucket` call.

<img src="media\fastapi.png" alt="media\fastapi.png">



Please get in touch with me at jamesysato@gmail.com if you would like to implement any of these features.
