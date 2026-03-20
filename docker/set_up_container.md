```
docker run -it -d \
-v /srv:/srv -v /mnt:/mnt -v /temp:/temp -v /home/eray/repo:/workspaces_data/repo -v /home/eray/project:/workspaces_data/project \
--ipc=host --env DISPLAY=$DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix --device /dev/dri --gpus=all -e NVIDIA_DRIVER_CAPABILITIES=all --restart unless-stopped \
-u $(id -u):$(id -g) \
-w /workspaces_data --name pytorch_2_3_1 565ac28ad01e /bin/bash
```

加``` -u $(id -u):$(id -g) ```第一次啟動
```
docker exec -u 0 -it cudaImage bash -c "groupadd -g $(id -g) eray && useradd -l -u $(id -u) -g eray -m eray && usermod -aG sudo eray"
```

```
# 以 root 身份進入已經在執行的容器
docker exec -u 0 -it pytorch_2_3_1 bash
```


```
docker create -v /workspaces_data --name data_volume_container -v /home/shared/:/workspaces_data/shared -v /home/eray/repo/:/workspaces_data/repo 
-v /home/eray/project/:/workspaces_data/project ubuntu:18.04
```