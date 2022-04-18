FROM tensorflow/tensorflow:latest-gpu

RUN pip install \
    absl-py>=0.9.0 \
    matplotlib>=3.0.0 \
    tensorflow-datasets>=3.0.0 \
    scipy \
    notebook

RUN mkdir /root/workspace
RUN mkdir /root/tensorflow_datasets
RUN chmod -R 777 /root

# run (locally) with:
# docker run --gpus all -p 8888:8888 --mount type=bind,source=$HOME/projects/o3d-nerf,target=/root/workspace --mount type=bind,source=$HOME/tensorflow_datasets,target=/root/tensorflow_datasets -it pmh47/slot_attention
# port forwarding (-p 8888:8888) is only required for jupyter

# can then use:
# PYTHONPATH=. python slot_attention/object_discovery/train.py  --model_dir slot_attention/checkpoints/object_discovery
# jupyter notebook --allow-root --ip=0.0.0.0 --no-browser
