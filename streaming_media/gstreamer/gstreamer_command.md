``` shell
gst-launch-1.0 -v rtspsrc location=rtsp://192.168.33.103:8554/C520WS latency=1000 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvideoconvert ! 'video/x-raw(memory:NVMM), format=NV12' ! nveglglessink

gst-launch-1.0 -v rtspsrc location=rtsp://127.0.0.1:8554/live protocols=tcp latency=200 ! rtph264depay ! h264parse ! nvv4l2decoder ! nveglglessink sync=false

```