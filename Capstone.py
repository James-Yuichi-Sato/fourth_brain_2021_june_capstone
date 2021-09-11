#!/usr/bin/env python
# coding: utf-8

# # To-Do List
# Database
# 
# Image Quality - How to check
# 
# GT quality - Mean pixels strength, pixel mean value
# 
# Images w/ metadata
# Metadata - Key Frames - Categories
# 
# Import batch of frames - Curate Frames - Check Ground Truth - Generate Metadata - Output
# WandB
# Flask Front End
# Image Output
# 
# scalability
# 
# wandb - monitor quality

# # Import FastAPI, FFmpeg, uvicorn, and JAAD

# In[1]:


get_ipython().run_cell_magic('capture', '', '!pip install fastapi ffmpeg uvicorn JAAD python-multipart tensorflow-gpu scikit-image imutils\n\nfrom fastapi import FastAPI, File, UploadFile\nimport nest_asyncio, uvicorn, os\n\nfrom io import BytesIO')


# # Import and Set Up WandB

# ## Import WandB Library

# In[2]:


get_ipython().run_cell_magic('capture', '', '!pip install wandb\nimport wandb')


# ## Log In to WandB

# In[3]:


wandb.login()


# # Set up ResNet50 Model

# In[4]:


from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

# img_path = 'elephant.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

# preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
# print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]


# # Set Up Google Cloud Parameters
# 

# ## Import Google Cloud Library

# In[5]:


from google.cloud import storage


# ## Set Up Google Cloud Project and Model Location

# In[6]:


project = 'mlops-content1' # Cloud Project Name
location = 'james-mlops-capstone' # Model Storage Bucket
model_dir = 'model'


# ## Create Storage Bucket

# In[7]:


storage_client = storage.Client.from_service_account_json('james-capstone-key.json')

bucket = storage_client.bucket(location)

# working_bucket = storage_client.bucket(location)


# ## Double Check Cloud Bucket

# In[8]:


get_ipython().run_cell_magic('capture', '', 'blobs = storage_client.list_blobs(location)\nfor blob in blobs:\n    print(blob.name)')


# # WandB Functions

# In[9]:


def init_wandb(project_name):
   global wandb_project
   wandb_project = str(project_name)
   wandb.init(project=wandb_project, sync_tensorboard=True)
   return True


# # Set File Location

# In[10]:


def set_folder_location(in_location):
    global location 
    location = str(in_location)
    global bucket
    bucket = storage_client.bucket(location)
    return True


# # Split Video to Frames and Upload

# ## Download Video to Local Instance

# In[11]:


def download_video(video_name):
    print("Downloading: " + str(video_name))
    blob = bucket.blob(video_name)
    blob.download_to_filename(video_name)


# ## Break down video to frames

# In[12]:


def split_video_frames(video_name):
    print("Splitting: " + str(video_name))
    folder = video_name[:-4]
    get_ipython().system('mkdir $folder')
    get_ipython().system('ffmpeg -i $video_name $folder/frame%04d.png -hide_banner -loglevel error')


# ## Upload Video Frames

# In[13]:


import glob

def upload_frames_from_folder(folder_name):
    folder_name = folder_name + "/*.png"
    print("Uploading Frames")
    for filename in glob.iglob(folder_name):
        print(filename, end="\r")
        blob = bucket.blob(filename)
        blob.upload_from_filename(filename)
    print("Done Uploading               ", end="\r")


# ## SSIM Compare Video Frames for Novel Frames

# ## Import Libraries

# In[14]:


# import the necessary packages
from skimage.metrics import structural_similarity as compare_ssim
import cv2


# ## Remove Blurry Images from Set

# ### Calculate Blurriness using Laplacian

# In[15]:


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


# ### Remove Blurry Images

# In[16]:


import numpy as np

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
    
    average_blur = np.mean(blurriness)
    print("Average Blur (Laplacian Variance): " + str(average_blur))
    blur_cutoff = average_blur*1.05 #+ ((1-average_blur)*0.1)
    print("Blur Cutoff (Laplacian Variance): " + str(blur_cutoff))
    
    print("Removing Blurry Images")
    
    for i in range(len(files)):
        if blurriness[i] > blur_cutoff:
            print("Deleting " + files[i] + " - Blurriness: " + str(blurriness[i]))
            os.remove(folder_name+'/'+files[i])
    print("Done Checking Frames                  ")


# # Deduplicate Similar Frames

# ## Calculate Similarity Between Images

# In[17]:


def compare_images(image1, image2):
    image_gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image_gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    diff, _ = compare_ssim(image_gray1, image_gray2, full=True)
    return diff


# ## Remove Duplicates

# In[18]:


def remove_duplicates(folder_name):
    files=sorted(os.listdir(str(folder_name)))
    files=files[1:]
    print("Removing Duplicate and Highly Similar Frames")
    for i in range(len(files)-1):
        #print(files[i])#, end="\r")
        print(files[i] + "             ", end="\r")
        image1 = cv2.imread(folder_name+'/'+files[i])
        image2 = cv2.imread(folder_name+'/'+files[i+1])
        diff = compare_images(image1, image2)
        if diff > 0.99:
            print("Deleting " + files[i] + " - Similarity: " + str(diff), end="\r")
            os.remove(folder_name+'/'+files[i])
    print("Done Checking Frames           ")


# # Full Video Analysis and Upload

# In[19]:


def clean_video(video_name):
    video_name = str(video_name)
    folder_name = str(video_name)[:-4]
    download_video(video_name)
    split_video_frames(video_name)
    remove_blurry_images(folder_name)
    remove_duplicates(folder_name)
    upload_frames_from_folder(folder_name)


# # Analyze Entire Bucket

# In[20]:


def clean_entire_bucket():
    files = get_ipython().getoutput('gsutil ls -r gs://$location')
    for file in files:
        name = file.strip("gs://"+location)
        print("Cleaning Video: " + str(name))
        clean_video(name)


# # Test Script

# In[21]:


#clean_video("video_0002.mp4")


# # Import Labels

# In[22]:


get_ipython().system('pip3 install pickle5')
import pickle5 as pickle
import pandas as pd

with open('jaad_database.pkl', 'rb') as pickle_file:
    pickle_data = pickle.load(pickle_file)

labels = pd.DataFrame(data=pickle_data)
labels['video_0001']


# # FastAPI Deployment

# In[23]:


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
    return {'message': ('Video: ' + str(string_input) + ' cleaned and uploaded to gs://' + location)}

@app.get('/clean_bucket')
async def full_clean():
    clean_entire_bucket()
    return {'message': ('Bucket: ' + location + ' cleaned and uploaded to gs://' + location)}

                
#@app.post('/predict_single')
#async def predict_api(file: UploadFile = File(...)):
#    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
#    if not extension:
#        return "Image must be jpg or png format!"
#    image = read_imagefile(await file.read())
#    prediction = run_predict_single(image)
#    prediction = str(prediction)
#    print(prediction)
#    return prediction


# # Run Deployment

# In[ ]:

if __name__ == '__main__':
  if "serve" in sys.argv: 
    nest_asyncio.apply()
    uvicorn.run(app, host='0.0.0.0', port=8000)


# In[ ]:




