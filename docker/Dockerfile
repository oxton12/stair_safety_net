FROM nvcr.io/nvidia/mxnet:24.06-py3

RUN rm -rf /usr/local/lib/python3.10/dist-packages/cv2
RUN pip install opencv-python==4.8.0.74
RUN apt update && apt install -y libgl1-mesa-dev

WORKDIR /workspace