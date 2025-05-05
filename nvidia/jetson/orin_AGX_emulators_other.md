1. jetson orin nano-8GB yolov8n640*640
    inf_avg_time: 0.0228
    infer FPS: 43.94703016818826
    total_avg_time: 0.0293
    FPS: 34.10212064222192
trtexec --loadEngine=/home/eray/tensorrt_test/model_hub/yolov8_640X640/yolov8n_onnx_change.engine --fp16 --warmUp=15 --verbose --workspace=4096 --streams=2
[04/09/2025-11:06:29] [I] 
[04/09/2025-11:06:29] [I] === Trace details ===
[04/09/2025-11:06:29] [I] Trace averages of 10 runs:
[04/09/2025-11:06:29] [I] Average on 10 runs - GPU latency: 16.794 ms - Host latency: 18.0551 ms (enqueue 2.0209 ms)
[04/09/2025-11:06:29] [I] 
[04/09/2025-11:06:29] [I] === Performance summary ===
[04/09/2025-11:06:29] [I] Throughput: 128.765 qps
[04/09/2025-11:06:29] [I] Latency: min = 8.06079 ms, max = 38.741 ms, mean = 15.5503 ms, median = 13.4062 ms, percentile(99%) = 38.741 ms
[04/09/2025-11:06:29] [I] Enqueue Time: min = 1.26172 ms, max = 3.33569 ms, mean = 1.89722 ms, median = 1.86401 ms, percentile(99%) = 3.33569 ms
[04/09/2025-11:06:29] [I] H2D Latency: min = 0.51709 ms, max = 1.81787 ms, mean = 0.68105 ms, median = 0.57959 ms, percentile(99%) = 1.81787 ms
[04/09/2025-11:06:29] [I] GPU Compute Time: min = 7.24829 ms, max = 36.7124 ms, mean = 14.4458 ms, median = 12.3491 ms, percentile(99%) = 36.7124 ms
[04/09/2025-11:06:29] [I] D2H Latency: min = 0.16626 ms, max = 1.0332 ms, mean = 0.42352 ms, median = 0.363281 ms, percentile(99%) = 1.0332 ms
[04/09/2025-11:06:29] [I] Total Host Walltime: 0.147555 s
[04/09/2025-11:06:29] [I] Total GPU Compute Time: 0.274469 s
[04/09/2025-11:06:29] [W] * GPU compute time is unstable, with coefficient of variance = 47.3653%.
[04/09/2025-11:06:29] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[04/09/2025-11:06:29] [I] Explanations of the performance metrics are printed in the verbose logs.
[04/09/2025-11:06:29] [V]
2. jetson orin nx-16GB yolov8n640*640
    inf_avg_time: 0.0179
    infer FPS: 55.99613070977472
    total_avg_time: 0.0232
    FPS: 43.16492003792844
[04/09/2025-14:48:59] [I] 
[04/09/2025-14:48:59] [I] === Trace details ===
[04/09/2025-14:48:59] [I] Trace averages of 10 runs:
[04/09/2025-14:48:59] [I] Average on 10 runs - GPU latency: 12.5305 ms - Host latency: 13.4482 ms (enqueue 1.25854 ms)
[04/09/2025-14:46:02] [I] 
[04/09/2025-14:46:02] [I] === Performance summary ===
[04/09/2025-14:46:02] [I] Throughput: 242.706 qps
[04/09/2025-14:46:02] [I] Latency: min = 7.02344 ms, max = 18.3943 ms, mean = 8.85544 ms, median = 8.35156 ms, percentile(99%) = 13.043 ms
[04/09/2025-14:46:02] [I] Enqueue Time: min = 0.863281 ms, max = 2.66406 ms, mean = 1.38301 ms, median = 1.55469 ms, percentile(99%) = 1.9668 ms
[04/09/2025-14:46:02] [I] H2D Latency: min = 0.21875 ms, max = 1.24414 ms, mean = 0.356999 ms, median = 0.359375 ms, percentile(99%) = 0.484375 ms
[04/09/2025-14:46:02] [I] GPU Compute Time: min = 6.55469 ms, max = 17.3673 ms, mean = 8.23807 ms, median = 7.73047 ms, percentile(99%) = 12.4238 ms
[04/09/2025-14:46:02] [I] D2H Latency: min = 0.132812 ms, max = 0.583008 ms, mean = 0.260376 ms, median = 0.25 ms, percentile(99%) = 0.367188 ms
[04/09/2025-14:46:02] [I] Total Host Walltime: 82.4003 s
[04/09/2025-14:46:02] [I] Total GPU Compute Time: 164.753 s
[04/09/2025-14:46:02] [W] * GPU compute time is unstable, with coefficient of variance = 13.5675%.
[04/09/2025-14:46:02] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[04/09/2025-14:46:02] [I] Explanations of the performance metrics are printed in the verbose logs.
[04/09/2025-14:46:02] [V]

2. jetson orin AGX-32GB yolov8n640*640
   1. batch : 1
        1. infer time:0.0053
        2. FPS:188 each run 1 frame

1. jetson orin AGX-32GB yolov8n1120*1120
    1. batch 1
        1. total_avg_time: 0.0197
        2. FPS: 50.88267402274194

## resnet18

model name:resnet18
infer time:0.0041
FPS:244 each run 1 frame

1. jetson orin AGX-32GB 3*224*224
### on GPU
   1. batch 1
      1. infer time 0.0011
      2. FPS 909
    [04/24/2025-13:04:46] [I] 
    [04/24/2025-13:04:46] [I] === Trace details ===
    [04/24/2025-13:04:46] [I] Trace averages of 10 runs:
    [04/24/2025-13:04:46] [I] Average on 10 runs - GPU latency: 0.982864 ms - Host latency: 1.02191 ms (enqueue 0.391095 ms)
    [04/24/2025-13:04:46] [I] 
    [04/24/2025-13:04:46] [I] === Performance summary ===
    [04/24/2025-13:04:46] [I] Throughput: 974.492 qps
    [04/24/2025-13:04:46] [I] Latency: min = 1.01416 ms, max = 4.14337 ms, mean = 1.06152 ms, median = 1.02237 ms, percentile(90%) = 1.02661 ms, percentile(95%) = 1.02954 ms, percentile(99%) = 2.11389 ms
    [04/24/2025-13:04:46] [I] Enqueue Time: min = 0.277504 ms, max = 0.66626 ms, mean = 0.307193 ms, median = 0.300903 ms, percentile(90%) = 0.326416 ms, percentile(95%) = 0.342651 ms, percentile(99%) = 0.384644 ms
    [04/24/2025-13:04:46] [I] H2D Latency: min = 0.0276184 ms, max = 0.0478516 ms, mean = 0.031765 ms, median = 0.0313721 ms, percentile(90%) = 0.0335083 ms, percentile(95%) = 0.0350342 ms, percentile(99%) = 0.0395508 ms
    [04/24/2025-13:04:46] [I] GPU Compute Time: min = 0.978149 ms, max = 4.10474 ms, mean = 1.02336 ms, median = 0.984375 ms, percentile(90%) = 0.987183 ms, percentile(95%) = 0.988159 ms, percentile(99%) = 2.07111 ms
    [04/24/2025-13:04:46] [I] D2H Latency: min = 0.00341797 ms, max = 0.00939941 ms, mean = 0.0063919 ms, median = 0.00624084 ms, percentile(90%) = 0.0078125 ms, percentile(95%) = 0.00805664 ms, percentile(99%) = 0.00854492 ms
    [04/24/2025-13:04:46] [I] Total Host Walltime: 2.9913 s
    [04/24/2025-13:04:46] [I] Total GPU Compute Time: 2.98311 s
    [04/24/2025-13:04:46] [W] * GPU compute time is unstable, with coefficient of variance = 28.6902%.
    [04/24/2025-13:04:46] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
    [04/24/2025-13:04:46] [I] Explanations of the performance metrics are printed in the verbose logs.
    [04/24/2025-13:04:46] [V]
   2. batch 2
      1. infer time:0.0013
      2. FPS:766 each run 2 frame
    [04/24/2025-16:39:34] [I] 
    [04/24/2025-16:39:34] [I] === Trace details ===
    [04/24/2025-16:39:34] [I] Trace averages of 10 runs:
    [04/24/2025-16:39:34] [I] Average on 10 runs - GPU latency: 1.12886 ms - Host latency: 1.19022 ms (enqueue 0.240723 ms)
    [04/24/2025-16:39:34] [I] 
    [04/24/2025-16:39:34] [I] === Performance summary ===
    [04/24/2025-16:39:34] [I] Throughput: 842.749 qps
    [04/24/2025-16:39:34] [I] Latency: min = 1.17212 ms, max = 4.28983 ms, mean = 1.24305 ms, median = 1.18677 ms, percentile(90%) = 1.19214 ms, percentile(95%) = 1.19629 ms, percentile(99%) = 3.99829 ms
    [04/24/2025-16:39:34] [I] Enqueue Time: min = 0.159424 ms, max = 0.473389 ms, mean = 0.180115 ms, median = 0.177246 ms, percentile(90%) = 0.187378 ms, percentile(95%) = 0.19873 ms, percentile(99%) = 0.238874 ms
    [04/24/2025-16:39:34] [I] H2D Latency: min = 0.0490723 ms, max = 0.0712891 ms, mean = 0.053557 ms, median = 0.0529785 ms, percentile(90%) = 0.0566406 ms, percentile(95%) = 0.0576172 ms, percentile(99%) = 0.0605774 ms
    [04/24/2025-16:39:34] [I] GPU Compute Time: min = 1.11548 ms, max = 4.23291 ms, mean = 1.1837 ms, median = 1.12744 ms, percentile(90%) = 1.13098 ms, percentile(95%) = 1.13397 ms, percentile(99%) = 3.93945 ms
    [04/24/2025-16:39:34] [I] D2H Latency: min = 0.00366211 ms, max = 0.0078125 ms, mean = 0.00578654 ms, median = 0.0057373 ms, percentile(90%) = 0.0067749 ms, percentile(95%) = 0.00708008 ms, percentile(99%) = 0.00744629 ms
    [04/24/2025-16:39:34] [I] Total Host Walltime: 3.00327 s
    [04/24/2025-16:39:34] [I] Total GPU Compute Time: 2.99595 s
    [04/24/2025-16:39:34] [W] * GPU compute time is unstable, with coefficient of variance = 29.1593%.
    [04/24/2025-16:39:34] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
    [04/24/2025-16:39:34] [I] Explanations of the performance metrics are printed in the verbose logs.
    [04/24/2025-16:39:34] [I]
   3. batch 4
       1. infer time:0.0022
       2. FPS:460 each run 4 frame
    [04/24/2025-16:42:08] [I] 
    [04/24/2025-16:42:08] [I] === Trace details ===
    [04/24/2025-16:42:08] [I] Trace averages of 10 runs:
    [04/24/2025-16:42:08] [I] Average on 10 runs - GPU latency: 1.96891 ms - Host latency: 2.06688 ms (enqueue 0.221352 ms)
    [04/24/2025-16:42:08] [I] 
    [04/24/2025-16:42:08] [I] === Performance summary ===
    [04/24/2025-16:42:08] [I] Throughput: 484.3 qps
    [04/24/2025-16:42:08] [I] Latency: min = 2.05188 ms, max = 5.15414 ms, mean = 2.16042 ms, median = 2.06213 ms, percentile(90%) = 2.0759 ms, percentile(95%) = 2.6095 ms, percentile(99%) = 5.07349 ms
    [04/24/2025-16:42:08] [I] Enqueue Time: min = 0.165527 ms, max = 0.324341 ms, mean = 0.184038 ms, median = 0.17981 ms, percentile(90%) = 0.197876 ms, percentile(95%) = 0.212433 ms, percentile(99%) = 0.245544 ms
    [04/24/2025-16:42:08] [I] H2D Latency: min = 0.0888672 ms, max = 0.118408 ms, mean = 0.0934762 ms, median = 0.0930176 ms, percentile(90%) = 0.0961914 ms, percentile(95%) = 0.098877 ms, percentile(99%) = 0.104187 ms
    [04/24/2025-16:42:08] [I] GPU Compute Time: min = 1.95483 ms, max = 5.05478 ms, mean = 2.06097 ms, median = 1.96301 ms, percentile(90%) = 1.97501 ms, percentile(95%) = 2.5072 ms, percentile(99%) = 4.97644 ms
    [04/24/2025-16:42:08] [I] D2H Latency: min = 0.00390625 ms, max = 0.00927734 ms, mean = 0.00597218 ms, median = 0.00598145 ms, percentile(90%) = 0.00683594 ms, percentile(95%) = 0.00708008 ms, percentile(99%) = 0.00732422 ms
    [04/24/2025-16:42:08] [I] Total Host Walltime: 3.0064 s
    [04/24/2025-16:42:08] [I] Total GPU Compute Time: 3.00078 s
    [04/24/2025-16:42:08] [W] * GPU compute time is unstable, with coefficient of variance = 22.0454%.
    [04/24/2025-16:42:08] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
    [04/24/2025-16:42:08] [I] Explanations of the performance metrics are printed in the verbose logs.
    [04/24/2025-16:42:08] [I]
   4. batch 8
       1. model name:resnet18_bs08
       2. infer time:0.0037
    [04/24/2025-16:51:05] [I] 
    [04/24/2025-16:51:05] [I] === Trace details ===
    [04/24/2025-16:51:05] [I] Trace averages of 10 runs:
    [04/24/2025-16:51:05] [I] Average on 10 runs - GPU latency: 3.42964 ms - Host latency: 3.63185 ms (enqueue 0.359261 ms)
    [04/24/2025-16:51:05] [I] 
    [04/24/2025-16:51:05] [I] === Performance summary ===
    [04/24/2025-16:51:05] [I] Throughput: 280.451 qps
    [04/24/2025-16:51:05] [I] Latency: min = 3.55298 ms, max = 7.41333 ms, mean = 3.74385 ms, median = 3.5752 ms, percentile(90%) = 3.60794 ms, percentile(95%) = 4.23877 ms, percentile(99%) = 7.13934 ms
    [04/24/2025-16:51:05] [I] Enqueue Time: min = 0.167999 ms, max = 0.480591 ms, mean = 0.201004 ms, median = 0.189667 ms, percentile(90%) = 0.231262 ms, percentile(95%) = 0.252319 ms, percentile(99%) = 0.341721 ms
    [04/24/2025-16:51:05] [I] H2D Latency: min = 0.159607 ms, max = 0.242976 ms, mean = 0.181237 ms, median = 0.180603 ms, percentile(90%) = 0.185425 ms, percentile(95%) = 0.188278 ms, percentile(99%) = 0.194878 ms
    [04/24/2025-16:51:05] [I] GPU Compute Time: min = 3.37085 ms, max = 7.22168 ms, mean = 3.55613 ms, median = 3.3877 ms, percentile(90%) = 3.41656 ms, percentile(95%) = 4.05078 ms, percentile(99%) = 6.95517 ms
    [04/24/2025-16:51:05] [I] D2H Latency: min = 0.00415039 ms, max = 0.00933838 ms, mean = 0.00647678 ms, median = 0.00634766 ms, percentile(90%) = 0.0078125 ms, percentile(95%) = 0.00830078 ms, percentile(99%) = 0.0090332 ms
    [04/24/2025-16:51:05] [I] Total Host Walltime: 3.01301 s
    [04/24/2025-16:51:05] [I] Total GPU Compute Time: 3.00493 s
    [04/24/2025-16:51:05] [W] * GPU compute time is unstable, with coefficient of variance = 18.8685%.
    [04/24/2025-16:51:05] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
    [04/24/2025-16:51:05] [I] Explanations of the performance metrics are printed in the verbose logs.
    [04/24/2025-16:51:05] [I]
   5. batch 10
       1. infer time:0.004
       2. FPS:246 each run 10 frame
    [04/24/2025-13:06:16] [I] 
    [04/24/2025-13:06:16] [I] === Trace details ===
    [04/24/2025-13:06:16] [I] Trace averages of 10 runs:
    [04/24/2025-13:06:16] [I] Average on 10 runs - GPU latency: 3.77243 ms - Host latency: 4.00366 ms (enqueue 0.244839 ms)
    [04/24/2025-13:06:16] [I] 
    [04/24/2025-13:06:16] [I] === Performance summary ===
    [04/24/2025-13:06:16] [I] Throughput: 253.873 qps
    [04/24/2025-13:06:16] [I] Latency: min = 3.9718 ms, max = 7.78174 ms, mean = 4.16452 ms, median = 4.00435 ms, percentile(90%) = 4.02893 ms, percentile(95%) = 4.64941 ms, percentile(99%) = 7.64355 ms
    [04/24/2025-13:06:16] [I] Enqueue Time: min = 0.175293 ms, max = 0.854004 ms, mean = 0.197893 ms, median = 0.186279 ms, percentile(90%) = 0.228333 ms, percentile(95%) = 0.247864 ms, percentile(99%) = 0.325439 ms
    [04/24/2025-13:06:16] [I] H2D Latency: min = 0.202148 ms, max = 0.241211 ms, mean = 0.225898 ms, median = 0.22583 ms, percentile(90%) = 0.229248 ms, percentile(95%) = 0.230713 ms, percentile(99%) = 0.237549 ms
    [04/24/2025-13:06:16] [I] GPU Compute Time: min = 3.74353 ms, max = 7.55017 ms, mean = 3.93162 ms, median = 3.77202 ms, percentile(90%) = 3.79138 ms, percentile(95%) = 4.41455 ms, percentile(99%) = 7.40625 ms
    [04/24/2025-13:06:16] [I] D2H Latency: min = 0.00415039 ms, max = 0.0101929 ms, mean = 0.00699988 ms, median = 0.00671387 ms, percentile(90%) = 0.00891113 ms, percentile(95%) = 0.00950623 ms, percentile(99%) = 0.00976562 ms
    [04/24/2025-13:06:16] [I] Total Host Walltime: 3.00938 s
    [04/24/2025-13:06:16] [I] Total GPU Compute Time: 3.00376 s
    [04/24/2025-13:06:16] [W] * GPU compute time is unstable, with coefficient of variance = 17.2817%.
    [04/24/2025-13:06:16] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
    [04/24/2025-13:06:16] [I] Explanations of the performance metrics are printed in the verbose logs.
    [04/24/2025-13:06:16] [V]
   6. batch 16
       1. infer time:0.0066
       2. FPS:151 each run 16 frame
    [04/24/2025-16:52:52] [I] 
    [04/24/2025-16:52:52] [I] === Trace details ===
    [04/24/2025-16:52:52] [I] Trace averages of 10 runs:
    [04/24/2025-16:52:52] [I] Average on 10 runs - GPU latency: 6.60569 ms - Host latency: 6.98299 ms (enqueue 0.277324 ms)
    [04/24/2025-16:52:52] [I] 
    [04/24/2025-16:52:52] [I] === Performance summary ===
    [04/24/2025-16:52:52] [I] Throughput: 153.637 qps
    [04/24/2025-16:52:52] [I] Latency: min = 6.48938 ms, max = 10.6628 ms, mean = 6.86432 ms, median = 6.55981 ms, percentile(90%) = 7.2081 ms, percentile(95%) = 9.86304 ms, percentile(99%) = 10.5693 ms
    [04/24/2025-16:52:52] [I] Enqueue Time: min = 0.178467 ms, max = 0.448944 ms, mean = 0.215454 ms, median = 0.200714 ms, percentile(90%) = 0.25769 ms, percentile(95%) = 0.286255 ms, percentile(99%) = 0.411072 ms
    [04/24/2025-16:52:52] [I] H2D Latency: min = 0.302612 ms, max = 0.45328 ms, mean = 0.362386 ms, median = 0.361526 ms, percentile(90%) = 0.368164 ms, percentile(95%) = 0.372803 ms, percentile(99%) = 0.384033 ms
    [04/24/2025-16:52:52] [I] GPU Compute Time: min = 6.16693 ms, max = 10.2942 ms, mean = 6.49387 ms, median = 6.18982 ms, percentile(90%) = 6.84093 ms, percentile(95%) = 9.49475 ms, percentile(99%) = 10.199 ms
    [04/24/2025-16:52:52] [I] D2H Latency: min = 0.00463867 ms, max = 0.0118408 ms, mean = 0.00807239 ms, median = 0.00805664 ms, percentile(90%) = 0.00952148 ms, percentile(95%) = 0.0100098 ms, percentile(99%) = 0.0107422 ms
    [04/24/2025-16:52:52] [I] Total Host Walltime: 3.0201 s
    [04/24/2025-16:52:52] [I] Total GPU Compute Time: 3.01315 s
    [04/24/2025-16:52:52] [W] * GPU compute time is unstable, with coefficient of variance = 14.249%.
    [04/24/2025-16:52:52] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
    [04/24/2025-16:52:52] [I] Explanations of the performance metrics are printed in the verbose logs.
    [04/24/2025-16:52:52] [I]
   7. batch 32
       1. infer time:0.0126
       2. FPS:79 each run 32 frame
    [04/24/2025-16:54:14] [I] 
    [04/24/2025-16:54:14] [I] === Trace details ===
    [04/24/2025-16:54:14] [I] Trace averages of 10 runs:
    [04/24/2025-16:54:14] [I] Average on 10 runs - GPU latency: 12.3089 ms - Host latency: 13.0574 ms (enqueue 0.28654 ms)
    [04/24/2025-16:54:14] [I] 
    [04/24/2025-16:54:14] [I] === Performance summary ===
    [04/24/2025-16:54:14] [I] Throughput: 79.7338 qps
    [04/24/2025-16:54:14] [I] Latency: min = 12.5095 ms, max = 17.108 ms, mean = 13.2058 ms, median = 12.605 ms, percentile(90%) = 16.3152 ms, percentile(95%) = 16.6784 ms, percentile(99%) = 17.0276 ms
    [04/24/2025-16:54:14] [I] Enqueue Time: min = 0.188477 ms, max = 0.39032 ms, mean = 0.232758 ms, median = 0.220627 ms, percentile(90%) = 0.283287 ms, percentile(95%) = 0.313843 ms, percentile(99%) = 0.384155 ms
    [04/24/2025-16:54:14] [I] H2D Latency: min = 0.629944 ms, max = 0.898656 ms, mean = 0.717973 ms, median = 0.717499 ms, percentile(90%) = 0.727005 ms, percentile(95%) = 0.730225 ms, percentile(99%) = 0.741882 ms
    [04/24/2025-16:54:14] [I] GPU Compute Time: min = 11.7996 ms, max = 16.3761 ms, mean = 12.4761 ms, median = 11.8749 ms, percentile(90%) = 15.5869 ms, percentile(95%) = 15.945 ms, percentile(99%) = 16.3005 ms
    [04/24/2025-16:54:14] [I] D2H Latency: min = 0.00634766 ms, max = 0.0147705 ms, mean = 0.0116794 ms, median = 0.0114441 ms, percentile(90%) = 0.0134583 ms, percentile(95%) = 0.0141602 ms, percentile(99%) = 0.0146484 ms
    [04/24/2025-16:54:14] [I] Total Host Walltime: 3.0351 s
    [04/24/2025-16:54:14] [I] Total GPU Compute Time: 3.01922 s
    [04/24/2025-16:54:14] [W] * GPU compute time is unstable, with coefficient of variance = 11.0137%.
    [04/24/2025-16:54:14] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
    [04/24/2025-16:54:14] [I] Explanations of the performance metrics are printed in the verbose logs.
    [04/24/2025-16:54:14] [I]
### on dla
   1. batch 1
      1. infer time:0.0086
      2. FPS:116 each run 1 frame
    [04/24/2025-17:47:15] [I] 
    [04/24/2025-17:47:15] [I] === Trace details ===
    [04/24/2025-17:47:15] [I] Trace averages of 10 runs:
    [04/24/2025-17:47:15] [I] Average on 10 runs - GPU latency: 8.3407 ms - Host latency: 8.37798 ms (enqueue 0.309539 ms)
    [04/24/2025-17:47:15] [I] 
    [04/24/2025-17:47:15] [I] === Performance summary ===
    [04/24/2025-17:47:15] [I] Throughput: 119.148 qps
    [04/24/2025-17:47:15] [I] Latency: min = 8.36548 ms, max = 11.2438 ms, mean = 8.42566 ms, median = 8.37842 ms, percentile(90%) = 8.52588 ms, percentile(95%) = 8.53589 ms, percentile(99%) = 9.32458 ms
    [04/24/2025-17:47:15] [I] Enqueue Time: min = 0.224365 ms, max = 0.433594 ms, mean = 0.270539 ms, median = 0.260071 ms, percentile(90%) = 0.304443 ms, percentile(95%) = 0.353027 ms, percentile(99%) = 0.409284 ms
    [04/24/2025-17:47:15] [I] H2D Latency: min = 0.026062 ms, max = 0.0415039 ms, mean = 0.0305558 ms, median = 0.0300751 ms, percentile(90%) = 0.0332031 ms, percentile(95%) = 0.03479 ms, percentile(99%) = 0.039917 ms
    [04/24/2025-17:47:15] [I] GPU Compute Time: min = 8.33215 ms, max = 11.2059 ms, mean = 8.38987 ms, median = 8.34277 ms, percentile(90%) = 8.48901 ms, percentile(95%) = 8.49948 ms, percentile(99%) = 9.29065 ms
    [04/24/2025-17:47:15] [I] D2H Latency: min = 0.00292969 ms, max = 0.00671387 ms, mean = 0.00524094 ms, median = 0.00512695 ms, percentile(90%) = 0.00610352 ms, percentile(95%) = 0.00622559 ms, percentile(99%) = 0.0065918 ms
    [04/24/2025-17:47:15] [I] Total Host Walltime: 3.02145 s
    [04/24/2025-17:47:15] [I] Total GPU Compute Time: 3.02035 s
    [04/24/2025-17:47:15] [W] * GPU compute time is unstable, with coefficient of variance = 2.84846%.
    [04/24/2025-17:47:15] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
    [04/24/2025-17:47:15] [I] Explanations of the performance metrics are printed in the verbose logs.
    [04/24/2025-17:47:15] [I]   
   2. batch 2
      1. infer time:0.0172
      2. FPS:29 each run 04_dla frame
    [04/24/2025-17:49:30] [I] 
    [04/24/2025-17:49:30] [I] === Trace details ===
    [04/24/2025-17:49:30] [I] Trace averages of 10 runs:
    [04/24/2025-17:49:30] [I] Average on 10 runs - GPU latency: 16.8985 ms - Host latency: 16.9488 ms (enqueue 0.288424 ms)
    [04/24/2025-17:49:30] [I] 
    [04/24/2025-17:49:30] [I] === Performance summary ===
    [04/24/2025-17:49:30] [I] Throughput: 59.0846 qps
    [04/24/2025-17:49:30] [I] Latency: min = 16.9014 ms, max = 19.5129 ms, mean = 16.9752 ms, median = 16.9132 ms, percentile(90%) = 17.066 ms, percentile(95%) = 17.0764 ms, percentile(99%) = 18.6882 ms
    [04/24/2025-17:49:30] [I] Enqueue Time: min = 0.244612 ms, max = 0.49292 ms, mean = 0.305841 ms, median = 0.294922 ms, percentile(90%) = 0.358398 ms, percentile(95%) = 0.422485 ms, percentile(99%) = 0.476562 ms
    [04/24/2025-17:49:30] [I] H2D Latency: min = 0.0393066 ms, max = 0.0744629 ms, mean = 0.0476229 ms, median = 0.0472412 ms, percentile(90%) = 0.0535889 ms, percentile(95%) = 0.0578613 ms, percentile(99%) = 0.0646973 ms
    [04/24/2025-17:49:30] [I] GPU Compute Time: min = 16.8521 ms, max = 19.4673 ms, mean = 16.9222 ms, median = 16.8605 ms, percentile(90%) = 17.0135 ms, percentile(95%) = 17.0209 ms, percentile(99%) = 18.6228 ms
    [04/24/2025-17:49:30] [I] D2H Latency: min = 0.00341797 ms, max = 0.00708008 ms, mean = 0.00534045 ms, median = 0.00524902 ms, percentile(90%) = 0.00634766 ms, percentile(95%) = 0.0065918 ms, percentile(99%) = 0.00695801 ms
    [04/24/2025-17:49:30] [I] Total Host Walltime: 3.02955 s
    [04/24/2025-17:49:30] [I] Total GPU Compute Time: 3.02907 s
    [04/24/2025-17:49:30] [W] * GPU compute time is unstable, with coefficient of variance = 1.46755%.
    [04/24/2025-17:49:30] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
    [04/24/2025-17:49:30] [I] Explanations of the performance metrics are printed in the verbose logs.
    [04/24/2025-17:49:30] [I]
   3. batch 4
      1. infer time:0.0348
      2. FPS:58 each run 02_dla frame
    [04/24/2025-17:52:15] [I] 
    [04/24/2025-17:52:15] [I] === Trace details ===
    [04/24/2025-17:52:15] [I] Trace averages of 10 runs:
    [04/24/2025-17:52:15] [I] Average on 10 runs - GPU latency: 34.4408 ms - Host latency: 34.5361 ms (enqueue 0.338582 ms)
    [04/24/2025-17:52:15] [I] 
    [04/24/2025-17:52:15] [I] === Performance summary ===
    [04/24/2025-17:52:15] [I] Throughput: 29.0079 qps
    [04/24/2025-17:52:15] [I] Latency: min = 34.4722 ms, max = 35.84 ms, mean = 34.5672 ms, median = 34.4889 ms, percentile(90%) = 34.6443 ms, percentile(95%) = 34.6688 ms, percentile(99%) = 35.84 ms
    [04/24/2025-17:52:15] [I] Enqueue Time: min = 0.262329 ms, max = 0.768188 ms, mean = 0.359358 ms, median = 0.343018 ms, percentile(90%) = 0.457428 ms, percentile(95%) = 0.47113 ms, percentile(99%) = 0.768188 ms
    [04/24/2025-17:52:15] [I] H2D Latency: min = 0.0765762 ms, max = 0.103333 ms, mean = 0.0908837 ms, median = 0.0910034 ms, percentile(90%) = 0.0960693 ms, percentile(95%) = 0.0966797 ms, percentile(99%) = 0.103333 ms
    [04/24/2025-17:52:15] [I] GPU Compute Time: min = 34.3812 ms, max = 35.7479 ms, mean = 34.4702 ms, median = 34.3911 ms, percentile(90%) = 34.5471 ms, percentile(95%) = 34.5703 ms, percentile(99%) = 35.7479 ms
    [04/24/2025-17:52:15] [I] D2H Latency: min = 0.00317383 ms, max = 0.0078125 ms, mean = 0.00603091 ms, median = 0.00610352 ms, percentile(90%) = 0.00708008 ms, percentile(95%) = 0.00732422 ms, percentile(99%) = 0.0078125 ms
    [04/24/2025-17:52:15] [I] Total Host Walltime: 3.06813 s
    [04/24/2025-17:52:15] [I] Total GPU Compute Time: 3.06785 s
    [04/24/2025-17:52:15] [I] Explanations of the performance metrics are printed in the verbose logs.
    [04/24/2025-17:52:15] [I]
   4. batch 8
       1. infer time:0.0708
       2. FPS:14 each run 08_dla frame
    [04/25/2025-09:09:08] [I] 
    [04/25/2025-09:09:08] [I] === Trace details ===
    [04/25/2025-09:09:08] [I] Trace averages of 10 runs:
    [04/25/2025-09:09:08] [I] Average on 10 runs - GPU latency: 70.3367 ms - Host latency: 70.5213 ms (enqueue 0.42186 ms)
    [04/25/2025-09:09:08] [I] 
    [04/25/2025-09:09:08] [I] === Performance summary ===
    [04/25/2025-09:09:08] [I] Throughput: 13.9007 qps
    [04/25/2025-09:09:08] [I] Latency: min = 70.4011 ms, max = 72.3069 ms, mean = 70.5667 ms, median = 70.5634 ms, percentile(90%) = 70.5825 ms, percentile(95%) = 70.5916 ms, percentile(99%) = 72.3069 ms
    [04/25/2025-09:09:08] [I] Enqueue Time: min = 0.28064 ms, max = 0.610046 ms, mean = 0.472563 ms, median = 0.480469 ms, percentile(90%) = 0.574951 ms, percentile(95%) = 0.582458 ms, percentile(99%) = 0.610046 ms
    [04/25/2025-09:09:08] [I] H2D Latency: min = 0.142143 ms, max = 0.187744 ms, mean = 0.179451 ms, median = 0.179932 ms, percentile(90%) = 0.185318 ms, percentile(95%) = 0.187012 ms, percentile(99%) = 0.187744 ms
    [04/25/2025-09:09:08] [I] GPU Compute Time: min = 70.2188 ms, max = 72.1212 ms, mean = 70.3803 ms, median = 70.3751 ms, percentile(90%) = 70.3949 ms, percentile(95%) = 70.4023 ms, percentile(99%) = 72.1212 ms
    [04/25/2025-09:09:08] [I] D2H Latency: min = 0.00390625 ms, max = 0.00830078 ms, mean = 0.00696072 ms, median = 0.00732422 ms, percentile(90%) = 0.0078125 ms, percentile(95%) = 0.00805664 ms, percentile(99%) = 0.00830078 ms
    [04/25/2025-09:09:08] [I] Total Host Walltime: 3.23724 s
    [04/25/2025-09:09:08] [I] Total GPU Compute Time: 3.16712 s
    [04/25/2025-09:09:08] [I] Explanations of the performance metrics are printed in the verbose logs.
    [04/25/2025-09:09:08] [I] 
   5. batch 10
       1. infer time:0.0889
       2. FPS:11 each run 10_dla frame
    [04/25/2025-09:10:32] [I] 
    [04/25/2025-09:10:32] [I] === Trace details ===
    [04/25/2025-09:10:32] [I] Trace averages of 10 runs:
    [04/25/2025-09:10:32] [I] Average on 10 runs - GPU latency: 88.4961 ms - Host latency: 88.721 ms (enqueue 0.511905 ms)
    [04/25/2025-09:10:32] [I] 
    [04/25/2025-09:10:32] [I] === Performance summary ===
    [04/25/2025-09:10:32] [I] Throughput: 11.0075 qps
    [04/25/2025-09:10:32] [I] Latency: min = 88.4329 ms, max = 89.9864 ms, mean = 88.6208 ms, median = 88.5887 ms, percentile(90%) = 88.6183 ms, percentile(95%) = 88.75 ms, percentile(99%) = 89.9864 ms
    [04/25/2025-09:10:32] [I] Enqueue Time: min = 0.280701 ms, max = 0.619629 ms, mean = 0.531363 ms, median = 0.540283 ms, percentile(90%) = 0.573608 ms, percentile(95%) = 0.598389 ms, percentile(99%) = 0.619629 ms
    [04/25/2025-09:10:32] [I] H2D Latency: min = 0.200317 ms, max = 0.232178 ms, mean = 0.220744 ms, median = 0.220764 ms, percentile(90%) = 0.225098 ms, percentile(95%) = 0.228516 ms, percentile(99%) = 0.232178 ms
    [04/25/2025-09:10:32] [I] GPU Compute Time: min = 88.2067 ms, max = 89.76 ms, mean = 88.3927 ms, median = 88.3589 ms, percentile(90%) = 88.3921 ms, percentile(95%) = 88.5232 ms, percentile(99%) = 89.76 ms
    [04/25/2025-09:10:32] [I] D2H Latency: min = 0.00415039 ms, max = 0.00952148 ms, mean = 0.00740136 ms, median = 0.00732422 ms, percentile(90%) = 0.00872803 ms, percentile(95%) = 0.00891113 ms, percentile(99%) = 0.00952148 ms
    [04/25/2025-09:10:32] [I] Total Host Walltime: 3.27049 s
    [04/25/2025-09:10:32] [I] Total GPU Compute Time: 3.18214 s
    [04/25/2025-09:10:32] [I] Explanations of the performance metrics are printed in the verbose logs.
    [04/25/2025-09:10:32] [I]
   6. batch 16
       1. infer time:0.1431
       2. FPS:7 each run 16_dla frame
    [04/25/2025-09:11:43] [I] 
    [04/25/2025-09:11:43] [I] === Trace details ===
    [04/25/2025-09:11:43] [I] Trace averages of 10 runs:
    [04/25/2025-09:11:43] [I] Average on 10 runs - GPU latency: 142.423 ms - Host latency: 142.776 ms (enqueue 0.586237 ms)
    [04/25/2025-09:11:43] [I] 
    [04/25/2025-09:11:43] [I] === Performance summary ===
    [04/25/2025-09:11:43] [I] Throughput: 7.02255 qps
    [04/25/2025-09:11:43] [I] Latency: min = 142.615 ms, max = 143.472 ms, mean = 142.747 ms, median = 142.719 ms, percentile(90%) = 142.744 ms, percentile(95%) = 142.836 ms, percentile(99%) = 143.472 ms
    [04/25/2025-09:11:43] [I] Enqueue Time: min = 0.319504 ms, max = 0.678711 ms, mean = 0.590552 ms, median = 0.611359 ms, percentile(90%) = 0.658691 ms, percentile(95%) = 0.673584 ms, percentile(99%) = 0.678711 ms
    [04/25/2025-09:11:43] [I] H2D Latency: min = 0.27565 ms, max = 0.388916 ms, mean = 0.355649 ms, median = 0.361511 ms, percentile(90%) = 0.368408 ms, percentile(95%) = 0.369629 ms, percentile(99%) = 0.388916 ms
    [04/25/2025-09:11:43] [I] GPU Compute Time: min = 142.32 ms, max = 143.104 ms, mean = 142.383 ms, median = 142.348 ms, percentile(90%) = 142.377 ms, percentile(95%) = 142.469 ms, percentile(99%) = 143.104 ms
    [04/25/2025-09:11:43] [I] D2H Latency: min = 0.00463867 ms, max = 0.00976562 ms, mean = 0.00810308 ms, median = 0.00805664 ms, percentile(90%) = 0.00952148 ms, percentile(95%) = 0.00976562 ms, percentile(99%) = 0.00976562 ms
    [04/25/2025-09:11:43] [I] Total Host Walltime: 3.27516 s
    [04/25/2025-09:11:43] [I] Total GPU Compute Time: 3.27482 s
    [04/25/2025-09:11:43] [I] Explanations of the performance metrics are printed in the verbose logs.
    [04/25/2025-09:11:43] [I] 
   7. batch 32
       1. infer time:0.2853
       2. FPS:4 each run 32_dla frame
    [04/25/2025-09:12:37] [I] 
    [04/25/2025-09:12:37] [I] === Trace details ===
    [04/25/2025-09:12:37] [I] Trace averages of 10 runs:
    [04/25/2025-09:12:37] [I] Average on 10 runs - GPU latency: 285.018 ms - Host latency: 285.738 ms (enqueue 0.702197 ms)
    [04/25/2025-09:12:37] [I] 
    [04/25/2025-09:12:37] [I] === Performance summary ===
    [04/25/2025-09:12:37] [I] Throughput: 3.50867 qps
    [04/25/2025-09:12:37] [I] Latency: min = 285.227 ms, max = 287.872 ms, mean = 285.684 ms, median = 285.418 ms, percentile(90%) = 286.303 ms, percentile(95%) = 287.872 ms, percentile(99%) = 287.872 ms
    [04/25/2025-09:12:37] [I] Enqueue Time: min = 0.411316 ms, max = 0.779297 ms, mean = 0.711446 ms, median = 0.75061 ms, percentile(90%) = 0.765381 ms, percentile(95%) = 0.779297 ms, percentile(99%) = 0.779297 ms
    [04/25/2025-09:12:37] [I] H2D Latency: min = 0.542572 ms, max = 0.739136 ms, mean = 0.714663 ms, median = 0.730103 ms, percentile(90%) = 0.738525 ms, percentile(95%) = 0.739136 ms, percentile(99%) = 0.739136 ms
    [04/25/2025-09:12:37] [I] GPU Compute Time: min = 284.668 ms, max = 287.129 ms, mean = 284.96 ms, median = 284.676 ms, percentile(90%) = 285.564 ms, percentile(95%) = 287.129 ms, percentile(99%) = 287.129 ms
    [04/25/2025-09:12:37] [I] D2H Latency: min = 0.00634766 ms, max = 0.0107422 ms, mean = 0.00959269 ms, median = 0.00979614 ms, percentile(90%) = 0.010376 ms, percentile(95%) = 0.0107422 ms, percentile(99%) = 0.0107422 ms
    [04/25/2025-09:12:37] [I] Total Host Walltime: 3.4201 s
    [04/25/2025-09:12:37] [I] Total GPU Compute Time: 3.41952 s
    [04/25/2025-09:12:37] [I] Explanations of the performance metrics are printed in the verbose logs.
    [04/25/2025-09:12:37] [I]


### glass 
1. jetson orin AGX-32GB 3*704*352
    1. batch : 01
        infer time:0.0117
        FPS:86 each run 1 frame
    2. batch : 02
        infer time:0.0220
        FPS:45 each run 02 frame
    3. batch : 04
        infer time:0.0429
        FPS:23 each run 04 frame
    4. batch : 08
        infer time:0.0840
        FPS:12 each run 08 frame
    5. batch : 10
        infer time:0.0840
        FPS:12 each run 08 frame
    6. batch : 16
        infer time:0.0840
        FPS:12 each run 08 frame
    7. batch : 32
        infer time:0.3288
        FPS:3 each run 32 frame
