FROM python:3.7
RUN pip install jupyter fastapi ffmpeg uvicorn JAAD python-multipart tensorflow-gpu scikit-image imutils ipython
COPY . .
WORKDIR .

# Install build utilities
RUN apt-get update && \
    apt-get install -y protobuf-compiler python3-pip python3-dev git && \
    apt-get -y upgrade

#Install Object Detection dependencies
RUN python3 -m pip install fastapi ffmpeg uvicorn python-multipart tensorflow-gpu scikit-image imutils wandb tensorflow_hub Pillow google-cloud google google-cloud-storage opencv-python

RUN pip3 install pillow

# Install Object Detection API library
RUN cd /opt && \
    git clone https://github.com/tensorflow/models && \
    cd models/research && \
    protoc object_detection/protos/*.proto --python_out=.
    
RUN cd $HOME && \
    git clone https://github.com/James-Yuichi-Sato/fourth_brain_2021_june_capstone.git && \
    cp -a fourth_brain_2021_june_capstone /opt/ && \
    chmod u+x /opt/fourth_brain_2021_june_capstone/Capstone.py
    
    
RUN python3 /opt/fourth_brain_2021_june_capstone/Capstone.py

ENV WANDB_API_KEY=$YOUR_API_KEY_HERE
    
# expose ports
EXPOSE 8000

#Command
CMD ["python3", "/opt/fourth_brain_2021_june_capstone/Capstone.py", "serve"]