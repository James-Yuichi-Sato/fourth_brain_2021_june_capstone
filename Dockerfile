FROM python:3.7
RUN pip install jupyter fastapi ffmpeg uvicorn JAAD python-multipart tensorflow-gpu scikit-image imutils
COPY . .
WORKDIR .

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["jupyter", "notebook", "./Capstone.ipynb","--port=7999", "--no-browser", "--ip=0.0.0.0", "--allow-root","--NotebookApp.token='pydata'"]
