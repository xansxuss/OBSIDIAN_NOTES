# Knowledge Index

## 00 Index

## 01 FOUNDATION

#### Computer Science Core
- algorithm
- mathematics
- network
- C&CPP
- RUST
- OS

## 02 AI_Systems

#### AI 主體
- deployment/
- multimodal/
- dataset_engine/
- model_optimization
- model_architecture/
- training/
- optimization/
- TensorRT
- ONNX

## 03 EDGE_AI
- architecture
	- async_pipeline
	- cpu_gpu_coordination
	- execution scheduling
	- graph_execution
	- lockfree
	- multi_stream_dispatch
	- memory reuse
	- pipeline_orchestrator
	- ring_buffer
	- resource_manager
	- stream_synchronization
	- scheduler
	- stream reuse
	- thread_pool
 - acceleration
	 - CUDA
	 - - inference_engine  
		- TensorRT  
		- ONNXRuntime  
		- tensorflow
		- pytorch
		- tf-lite
	 - TVM
	 - OpenCL
	 - Vulkan_compute
 - deployment
 - memory
	 - allocator
	 - arena_allocator
	 - cache_alignment
	 - dma_buffer
	 - memory_pool
	 - pinned_memory
	 - shared_memory
	 - unified_memory
	 - zero_copy
	 - EGLImage
	 - NvBufSurface
	 - 
 - media_pipelines
	 - camera
	 - decode
	 - encode
	 - ffmpeg
	 - gstreamer
	 - multi_stream
	 - rtsp
	 - SDL
 - platforms
	 - ARM64
	 - Jetson
	 - Rockchip
	 - RockchipRaspberryPi
	 - X86
  - optimization
	  - benchmark
	  - kernel_fusion
	  - low_flops
	  - latency
	  - memory_bandwidth
	  - occupancy
	  - operator_fusion
	  - pruning
	  - profiling
	  - quantization
	  - tensor_layout
	  - throughput
	  - warp_efficiency
- runtime_optimization
 
 

## 04 COMPUTER_VISION

- 3d_vision
- detection
- imagsProcess
- segmentation
- tracking

## 05 STREAMING_MEDIA

- codec
- low_latency_pipeline
- EGLImage
- FFmpeg
- NVDEC
- RTSP
- SDL2

## 06 INFRASTRUCTURE

- container  
- backend  
- git_data

## 07 EMBEDDED

## 08 Research
- edge AI roadmap
- low FLOPs inference

## 09 PROJECTS

- anomaly_detection_system
- edge_surveillance
- multi_rtsp_inference
- UAV_AI
## 10.MATH

## 11 EXPERIMENTS
- benchmark
- failed
- profiling
- temporary_tests
- weird_behavior
## 12 Failed_idea

## 13 **SYSTEMS_INTELLIGENCE**
把 perception / compute / time / hardware / control 混合在一起的設計思考
#### - perception 系統架構
- event-based vision
- hybrid frame-event systems
	- event_based_tracking
- multi-rate sensing
#### - 時間系統（核心）
- continuous-time estimation
- latency-aware design
- asynchronous pipeline
#### - compute 架構
- CUDA streaming graph
- zero-copy pipeline
- multi-rate inference
#### - sensing fusion
- camera + IMU
- event + frame fusion
- sensor scheduling
#### - control loop
- tracking → prediction → actuation
- closed-loop latency compensation
