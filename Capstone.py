# Import Libraries
from fastapi import FastAPI, File, UploadFile
from skimage.metrics import structural_similarity as compare_ssim

import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
from PIL import Image
from utils import label_map_util
import ffmpeg, shutil

from google.cloud import storage
import nest_asyncio, uvicorn, os, pathlib

import cv2, wandb

project = 'mlops-content1' # Cloud Project Name
location = 'james-mlops-capstone' # Model Storage Bucket

storage_client = storage.Client.from_service_account_json('gcp-key.json')
bucket = storage_client.bucket(location)

def init_wandb(project_name):
    global wandb_project
    wandb_project = str(project_name)
    wandb.init(project=wandb_project, sync_tensorboard=True)
    return True
   
def set_folder_location(in_location):
    global location 
    location = str(in_location)
    global bucket
    bucket = storage_client.bucket(location)
    return True

def download_video(video_name):
    print("Downloading: " + str(video_name))
    blob = bucket.blob(video_name)
    blob.download_to_filename(video_name)
    
def split_video_frames(video_name):
    print("Splitting: " + str(video_name))
    folder = video_name[:-4]
    try:
        shutil.rmtree(str(folder))
    except:
        pass
    os.mkdir(str(folder))
    
    video_capture = cv2.VideoCapture(str(video_name))
    saved_frame_name = 1

    while True:
        print("Frame: " + format(saved_frame_name, '05d'), end="\r")
        success, frame = video_capture.read()

        if success:
            cv2.imwrite(f"{str(folder)}/frame{format(saved_frame_name, '05d')}.png", frame)
            saved_frame_name += 1
        else:
            break
    print("Done                       ")

def upload_frames_from_folder(folder_name):
    files=sorted(os.listdir(str(folder_name)))
    #files=files[1:]
    
    print("Uploading Frames")
    for i in range(len(files)):
        print(files[i] + "             ", end="\r")
        blob = bucket.blob(folder_name + "/" + files[i])
        blob.upload_from_filename(folder_name + "/" + files[i])
        
    print("Done Uploading               ", end="\r")
    
def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

def remove_blurry_images(folder_name):
    files=sorted(os.listdir(str(folder_name)))
    files=files[1:]
    
    blurriness = np.zeros(len(files))
    
    print("Calculating Average Blurriness")
    for i in range(len(files)):
        print(files[i] + "             ", end="\r")
        img=cv2.imread(folder_name+'/'+files[i])
        img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurriness[i]=variance_of_laplacian(img_gray)
        wandb.log({'Individual Laplacian': blurriness[i]})
    
    median_blur = np.median(blurriness)
    wandb.log({'Batch Median Laplacian': median_blur})
    print("Median Blur (Laplacian Variance): " + str(median_blur))
    blur_cutoff = median_blur*1.05 #+ ((1-average_blur)*0.1)
    print("Blur Cutoff (Laplacian Variance): " + str(blur_cutoff))
    
    print("Removing Noisy Images")
    
    count = 0
    for i in range(len(files)):
        if blurriness[i] > blur_cutoff:
            print("Deleting " + files[i] + " - Laplacian Noisiness: " + str(blurriness[i]))
            os.remove(folder_name+'/'+files[i])
            count += 1
    blur_ratio = count/len(files)
    wandb.log({'Noisy Frame Ratio': blur_ratio})
    print("Done Checking Frames                  ")
    
def compare_images(image1, image2):
    image_gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image_gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    diff, _ = compare_ssim(image_gray1, image_gray2, full=True)
    return diff
    
def remove_duplicates(folder_name):
    files=sorted(os.listdir(str(folder_name)))
    files=files[1:]
    print("Removing Duplicate and Highly Similar Frames\nCalculating Frame Similarities")
    
    diff = np.zeros(len(files)-1)    
    
    for i in range(len(files)-1):
        image1 = cv2.imread(folder_name+'/'+files[i])
        image2 = cv2.imread(folder_name+'/'+files[i+1])
        diff[i] = compare_images(image1, image2)
        wandb.log({'Individual Frame Similarities': diff[i]})
        print(str(diff[i]), end="\r")
    
    median_diff = np.median(diff)
    wandb.log({'Batch Median Frame Similarity': median_diff})
    
    diff_cutoff = median_diff*1.05
    
    if diff_cutoff < 0.95:
        diff_cutoff = 0.95
        
    print("Similarity Cutoff (OpenCV Compare Images): " + str(diff_cutoff))
    print("Removing Duplicate Images")
    
    count = 0
    for i in range(len(diff)):
        if diff[i] > 0.99:
            print("Deleting " + files[i] + " - Similarity: " + str(diff[i]), end="\r")
            os.remove(folder_name+'/'+files[i])
            wandb.log({'Duplicates Similarity': diff})
            count += 1
        
    duplicate_ratio = count/len(files)
    wandb.log({'Batch Duplicate Remove Ratio': duplicate_ratio})
    print("Done Checking Frames, " + str(count) + " frames removed.")
    
model_url = 'http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz'
base_url = os.path.dirname(model_url)+"/"
model_file = os.path.basename(model_url)
model_name = os.path.splitext(os.path.splitext(model_file)[0])[0]
model_dir = tf.keras.utils.get_file(fname=model_name, origin=base_url + model_file, untar=True)
model_dir = pathlib.Path(model_dir)/"saved_model"
model = tf.saved_model.load(str(model_dir))
model = model.signatures['serving_default']

#label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
#category_index = label_map_util.create_category_index(categories)

def detect_objects(image):
    img = Image.open(file_)
    input_tensor = tf.convert_to_tensor(img)
    input_tensor = input_tensor[tf.newaxis,...]
    output_dict = model(input_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() for key,value in output_dict.items()}
    classes = output_dict['detection_classes'].astype(np.int64)
    wandb.log({'Detection per Image': num_detections})
    return classes

def detect_file(folder_name):
    for file in sorted(os.listdir(folder_name)):
        detect_objects(file)
        with open("{folder_name}/{file}.txt", "w") as text_file:
            text_file.write(str(classes))


def clean_video(video_name):
    video_name = str(video_name)
    folder_name = str(video_name)[:-4]
    download_video(video_name)
    split_video_frames(video_name)
    remove_blurry_images(folder_name)
    remove_duplicates(folder_name)
    detect_file(folder_name)
    upload_frames_from_folder(folder_name)
    shutil.rmtree(folder_name)
    os.remove(video_name)

def clean_entire_bucket():
    blobs = storage_client.list_blobs(location)
    for blob in blobs:
        clean_video(blob.name)

app = FastAPI()

@app.on_event("startup")
def start_wandb():
    init_wandb(location)
    return {'message': ('Weights and Balances Started as project: ' + wandb_project)}

@app.get('/')
def index():
    return {'message': 'This is the homepage of the model, add \'/docs\' to the end of the URL to access FastAPI to make predictions with the model'}

@app.get('/set_gcp_location')
def set_gcp_location(string_input):
    set_folder_location(str(string_input))
    return {'message': ('GCP Location Set to: ' + location)}

@app.get('/clean_single_video')
async def single_clean(string_input):
    clean_video(str(string_input))
    return {'message': ('Video: ' + str(string_input) + ' cleaned and uploaded to gs://' + location + "/" + str(string_input))}

@app.get('/clean_bucket')
async def full_clean():
    clean_entire_bucket()
    return {'message': ('Bucket: ' + location + ' cleaned and uploaded to gs://' + location)}

if __name__ == '__main__':
  if "serve" in sys.argv: 
    nest_asyncio.apply()
    wandb.login(relogin=True)
    uvicorn.run(app, host='0.0.0.0', port=8080)