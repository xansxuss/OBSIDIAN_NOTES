```bash
推送串流
ffmpeg -hwaccel cuda -i /home/eray/repo/media/Night_Walk_in_Tokyo_Shibuya_4k_h264.mp4 -c:v h264_nvenc -pix_fmt yuv420p -preset p2 -tune ll -g 30 -b:v 10M -maxrate:v 12M -bufsize:v 20M -rc cbr -spatial-aq 1 -f rtsp rtsp://localhost:8554/live

ffmpeg -re -stream_loop -1 -i /workspaces_data/repo/media/Night_Walk_in_Tokyo_Shibuya_4k_h264.mp4 -c:v h264_nvenc -preset p6 -spatial-aq 1 -rc vbr -cq 24 -b:v 5M -maxrate:v 8M -bufsize:v 16M -delay 0 -bf 0 -g 60 -pix_fmt yuv420p -rtsp_transport tcp -f rtsp rtsp://localhost:${RTSP_PORT}/test


接收串流
ffplay -vcodec h264_cuvid -rtsp_transport tcp rtsp://192.168.33.103:8554/C520WS
```