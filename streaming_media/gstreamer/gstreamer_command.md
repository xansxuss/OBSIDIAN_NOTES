``` shell
gst-launch-1.0 -v rtspsrc location=rtsp://192.168.33.103:8554/C520WS latency=1000 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvideoconvert ! 'video/x-raw(memory:NVMM), format=NV12' ! nveglglessink

gst-launch-1.0 -v rtspsrc location=rtsp://127.0.0.1:8554/live protocols=tcp latency=200 ! rtph264depay ! h264parse ! nvv4l2decoder ! nveglglessink sync=false

x86:
gst-launch-1.0 -v rtspsrc location=rtsp://192.168.33.76:8554/test latency=200 protocols=tcp ! rtph264depay ! h264parse ! nvh264dec ! nvvideoconvert ! video/x-raw,format=NV12 ! fakesink sync=false

gst-launch-1.0 -v rtspsrc location=rtsp://192.168.33.76:8554/test latency=200 protocols=tcp ! rtph264depay ! h264parse ! nvh264dec ! nvvideoconvert ! video/x-raw,format=NV12 ! nveglglessink

gst-launch-1.0 -v rtspsrc location=rtsp://192.168.33.76:8554/test latency=200 protocols=tcp ! \ rtph264depay ! \ h264parse ! \ nvh264dec ! \ queue max-size-buffers=16 max-size-bytes=0 max-size-time=0 ! \ nvvideoconvert ! \ "video/x-raw(memory:NVMM), format=NV12" ! \ nveglglessink sync=false

jetson:
gst-launch-1.0 rtspsrc location=rtsp://192.168.33.184:8554/oil_1 protocols=tcp ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! 'video/x-raw, format=I420' ! nv3dsink

```