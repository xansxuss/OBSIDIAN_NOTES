gst-launch-1.0 -v rtspsrc location=rtsp://192.168.33.76:8554/live latency=1000 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvideoconvert ! 'video/x-raw(memory:NVMM), format=NV12' ! nveglglessink

gst-launch-1.0 -v rtspsrc location=rtsp://127.0.0.1:8554/live protocols=tcp latency=200 ! rtph264depay ! h264parse ! nvv4l2decoder ! nveglglessink sync=false


推送串流
ffmpeg -hwaccel cuda -i /home/eray/repo/media/Night_Walk_in_Tokyo_Shibuya_4k_h264.mp4 -c:v h264_nvenc -pix_fmt yuv420p -preset p2 -tune ll -g 30 -b:v 10M -maxrate:v 12M -bufsize:v 20M -rc cbr -spatial-aq 1 -f rtsp rtsp://localhost:8554/live
ffmpeg -re -stream_loop -1 -i /home/eray/repo/media/Night_Walk_in_Tokyo_Shibuya_4k_h264.mp4 -c copy -bsf:v h264_mp4toannexb -f rtsp -max_interleave_delta 0 rtsp://localhost:$RTSP_PORT/live


接收串流
ffplay -vcodec h264_cuvid -rtsp_transport tcp rtsp://localhost:8554/live