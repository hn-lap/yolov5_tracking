FROM ubuntu:latest
WORKDIR /yolov5_tracking
COPY . /yolov5_tracking
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential hdf5-tools libgl1 libgtk2.0-dev
RUN pip install -r setup.txt
CMD /bin/sh
