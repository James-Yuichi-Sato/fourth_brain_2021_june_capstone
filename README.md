# Smart Annotation Pipeline for Video Sequences

## Fourth Brain 2021 June MLOps Cohort Capstone

## Project Description

In this project, our aim is to improve the video annotation workflow by removing redundant frames in video, as well as removing noisy frames and gaining insights into the videos using ML model detection as well as the metrics used to remove noisy and duplicate frames.

We developed the workflow using the dashcam videos openly available in the [JAAD Dataset](https://github.com/ykotseruba/JAAD).

This workflow is intended to provide insight to improve the data extraction - data validation - data preparation stages in an MLOps workflow.

<img src="media\mlops-continuous-delivery-and-automation-pipelines-in-machine-learning-3-ml-automation-ct.png" alt="media\mlops-continuous-delivery-and-automation-pipelines-in-machine-learning-3-ml-automation-ct.png"></img>

https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning



## Expected Output

This software returns pared down frames that are within difference and noisiness threshold calculated during runtime in `.png` format as well as metrics calculated during runtime in `.yaml` format.

### YAML Data

The `yaml` file contains the following metadata of the video:

- Name of Video File `yaml` was generated from
- Number of original frames in the Video File when first split
- Number of frames removed during the deblurring filter
- Number of frames removed during the deduplicating filter
- Minimum, maximum, and median Laplacian of all frames of video
- Ratio and absolute number of frames removed during the Laplacian filter
- Median frame structural similarity during deduplication filter
- Ratio and absolute number of frames removed during the deduplication filter
- Number of detected objects and classification counts of detected objects on the filtered frames by frame

**Examples of the Output File can be found [HERE]()**

<img src="media\yaml.png" alt="media\yaml.png"></img>

Example of `yaml` output

## Components of Workflow

### Overview

<img src="media\Workflow.drawio.png" alt="media\Workflow.drawio.png"></img>







### Data Storage

This workflow uses the GCP Cloud Storage library to bulk store the video footage for ingest, and exports the filtered frames and yaml metadata outputs to a subfolder named by video. The data is stored on the local machine during runtime and cleared from the local machine after uploaded back to GCP.



Before:

<img src="media\GCP_before.png" alt="media\GCP_before.png"></img>



After:

<img src="media\GCP_after.png" alt="media\GCP_after.png"></img>

### Frame Separation

Software separates frames from videos using `OpenCV`. As long as the `OpenCV` supports the video format used, there should be no issue with the software stack. The frames are stored in a folder with the same name as the video.

### Noisy Frame Reduction

The software calculates the Laplacian variance of all of the frames of the video using OpenCV, then removes frames with 5% more Laplacian variance than the median. This value was chosen as it would prevent video with multiple types of scenes removing frames due to a scene change rather than noisiness. This function removed between 0% and 45% of frames depending on the video in the testing dataset.

### Similar Frame Reduction

The software uses the structural similarity function from OpenCV to compare frames. It first calculates the similarity between each frame sequentially first, than delete frames that are more than 95% or 105% of the median similarity similar, whichever is greater. This function removed between 0% and 60% of frames depending on the video in the testing dataset.

### Metadata Tagging for Storage and Retrieval

The software uses the open-source [Faster RCNN Resnet50 model trained on Coco dataset](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz) to classify objects in each frame. The classifications per frame are saved in the video's yaml metadata file with the same name as the video file.

<img src="media\GCP_frames_yaml.png" alt="media\GCP_frames_yaml.png"></img>

## Running the Software

### Weights and Biases

This software uses Weights and Biases to show real time values from the analysis. Create the API key necessary to connect the software at wandb.ai/authorize

<img src="media\weights_and_biases3.png" alt="media\weights_and_biases3.png"></img>

### GCP IAM Key

The software uses a `gcp_key.json` file to access the google cloud storage instance. You can create the file on [Google Cloud](https://cloud.google.com/iam/docs/creating-managing-service-account-keys).

### FastAPI

This software exposes FastAPI on port 8000 of the machine. You can access all of the functions using `http://<EXTERNAL_URL>:8000/doc/`. You can frame clean individual videos in the GCP bucket using the `clean_single_video` call. You can frame clean all videos in the GCP bucket using the `clean_bucket` call.

<img src="media\fastapi.png" alt="media\fastapi.png"></img>



Please get in touch with me at jamesysato@gmail.com if you would like to implement any of these features.
