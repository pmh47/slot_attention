FROM tensorflow/tensorflow:latest-gpu

RUN pip install \
    absl-py>=0.9.0 \
    matplotlib>=3.0.0 \
    tensorflow-datasets>=3.0.0 \
    scipy

RUN mkdir /root/workspace
RUN mkdir /root/tensorflow_datasets

# run with:
# docker run --gpus all --mount type=bind,source=$HOME/projects/o3d-nerf,target=/root/workspace --mount type=bind,source=$HOME/tensorflow_datasets,target=/root/tensorflow_datasets -it pmh47/slot_attention
