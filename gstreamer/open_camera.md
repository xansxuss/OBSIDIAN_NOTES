from camera read stream and write to file (mjpeg)
``` gst-launch-1.0 -e v4l2src device="/dev/video0" ! "image/jpeg,width=1920,height=1080" ! jpegdec ! videoconvert ! x264enc bitrate=2048 ! mp4mux ! filesink location=/home/eray/output.mp4 ```

from camera read strem and show on window

``` gst-launch-1.0 -e v4l2src device="/dev/video0" ! "image/jpeg,width=1920,height=1080" ! jpegdec ! videoconvert ! autovideosink ```

from camera get frame and push to rtsp server

``` gst-launch-1.0 -e v4l2src device="/dev/video5" ! "image/jpeg,width=1920,height=1080,framerate=30/1" ! jpegdec ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=ultrafast ! rtspclientsink location=rtsp://192.168.33.176:8554/mystream ```

hardware speedup ``` gst-launch-1.0 -e -v v4l2src device=/dev/video17 ! video/x-raw,width=3840,height=2160 ! v4l2h264enc extra-controls="cid,video_gop_size=30" capture-io-mode=mmap ! rtspclientsink location=rtsp://localhost:8554/mystream ```

from rtsp camera get frame and push to rtsp server

``` gst-launch-1.0 rtspsrc location="rtsp://admin:eray80661707@192.168.33.15:554/channel=1&stream=0.sdp" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert !  x264enc tune=zerolatency bitrate=3000 speed-preset=ultrafast ! rtspclientsink location=rtsp://192.168.33.176:8554/mystream ```

hardware speedup ``` gst-launch-1.0 rtspsrc location="rtsp://admin:eray80661707@192.168.33.15:554/channel=1&stream=0.sdp" ! rtph264depay ! h264parse ! v4l2h264dec ! videoconvert ! v4l2h264enc extra-controls="cid,video_gop_size=30" capture-io-mode=mmap ! rtspclientsink location=rtsp://192.168.33.125:8554/mystream ```