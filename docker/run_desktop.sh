#! /bin/bash

docker run --rm -it \
	--net=host \
    --privileged \
	--runtime nvidia \
	--user $(id -u) \
	--env=NVIDIA_DRIVER_CAPABILITIES=all \
	--env=DISPLAY \
	--env=QT_X11_NO_MITSHM=1 \
	--pid=host \
	--cap-add=SYS_ADMIN \
	--cap-add=SYS_PTRACE \
	-e XDG_RUNTIME_DIR=/run/user/1000 \
	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	-v ~/.docker_bash_history:/home/admin/.bash_history:rw \
	-v ~/.tmux.conf:/home/admin/.tmux.conf \
	-v $PWD/../:/SIFT_SLAM3:rw \
	-v /mnt/d/UWslam_dataset:/data:ro \
	-v /mnt/water:/media/water:ro \
	-v ~/.docker_bash_history:/root/.bash_history \
	-w /SIFT_SLAM3 \
	--name SIFT_SLAM3 \
	gidobot:knfu_slam_2204_humble \
	bash

# -v /media/kraft/af7cd17b-9563-477c-be1d-89aa6b8aebb6/data:/data:rw \
