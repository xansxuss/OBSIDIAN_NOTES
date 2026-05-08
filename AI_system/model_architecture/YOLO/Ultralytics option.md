### Train
#### 1. Train Settings

| Argument          | Type                     | Default  | Description                                                                                                                                                                                                                                                                             |
| ----------------- | ------------------------ | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `amp`             | `bool`                   | `True`   | Enables Automatic [Mixed Precision](https://www.ultralytics.com/glossary/mixed-precision) (AMP) training, reducing memory usage and possibly speeding up training with minimal impact on accuracy.                                                                                      |
| `batch`           | `int` or `float`         | `16`     | [Batch size](https://www.ultralytics.com/glossary/batch-size), with three modes: set as an integer (e.g., `batch=16`), auto mode for 60% GPU memory utilization (`batch=-1`), or auto mode with specified utilization fraction (`batch=0.70`).                                          |
| `box`             | `float`                  | `7.5`    | Weight of the box loss component in the [loss function](https://www.ultralytics.com/glossary/loss-function), influencing how much emphasis is placed on accurately predicting [bounding box](https://www.ultralytics.com/glossary/bounding-box) coordinates.                            |
| `cache`           | `bool`                   | `False`  | Enables caching of dataset images in memory (`True`/`ram`), on disk (`disk`), or disables it (`False`). Improves training speed by reducing disk I/O at the cost of increased memory usage.                                                                                             |
| `classes`         | `list[int]`              | `None`   | Specifies a list of class IDs to train on. Useful for filtering out and focusing only on certain classes during training.                                                                                                                                                               |
| `close_mosaic`    | `int`                    | `10`     | Disables mosaic [data augmentation](https://www.ultralytics.com/glossary/data-augmentation) in the last N epochs to stabilize training before completion. Setting to 0 disables this feature.                                                                                           |
| `cls`             | `float`                  | `0.5`    | Weight of the classification loss in the total loss function, affecting the importance of correct class prediction relative to other components.                                                                                                                                        |
| `compile`         | `bool` or `str`          | `False`  | Enables PyTorch 2.x `torch.compile` graph compilation with `backend='inductor'`. Accepts `True` → `"default"`, `False` → disables, or a string mode such as `"default"`, `"reduce-overhead"`, `"max-autotune-no-cudagraphs"`. Falls back to eager with a warning if unsupported.        |
| `cos_lr`          | `bool`                   | `False`  | Utilizes a cosine [learning rate](https://www.ultralytics.com/glossary/learning-rate) scheduler, adjusting the learning rate following a cosine curve over epochs. Helps in managing learning rate for better convergence.                                                              |
| `data`            | `str`                    | `None`   | Path to the dataset configuration file (e.g., `coco8.yaml`). This file contains dataset-specific parameters, including paths to training and [validation data](https://www.ultralytics.com/glossary/validation-data), class names, and number of classes.                               |
| `deterministic`   | `bool`                   | `True`   | Forces deterministic algorithm use, ensuring reproducibility but may affect performance and speed due to the restriction on non-deterministic algorithms.                                                                                                                               |
| `device`          | `int` or `str` or `list` | `None`   | Specifies the computational device(s) for training: a single GPU (`device=0`), multiple GPUs (`device=[0,1]`), CPU (`device=cpu`), MPS for Apple silicon (`device=mps`), or auto-selection of most idle GPU (`device=-1`) or multiple idle GPUs (`device=[-1,-1]`)                      |
| `dfl`             | `float`                  | `1.5`    | Weight of the distribution focal loss, used in certain YOLO versions for fine-grained classification.                                                                                                                                                                                   |
| `dropout`         | `float`                  | `0.0`    | Dropout rate for regularization in classification tasks, preventing overfitting by randomly omitting units during training.                                                                                                                                                             |
| `epochs`          | `int`                    | `100`    | Total number of training epochs. Each [epoch](https://www.ultralytics.com/glossary/epoch) represents a full pass over the entire dataset. Adjusting this value can affect training duration and model performance.                                                                      |
| `exist_ok`        | `bool`                   | `False`  | If True, allows overwriting of an existing project/name directory. Useful for iterative experimentation without needing to manually clear previous outputs.                                                                                                                             |
| `fraction`        | `float`                  | `1.0`    | Specifies the fraction of the dataset to use for training. Allows for training on a subset of the full dataset, useful for experiments or when resources are limited.                                                                                                                   |
| `freeze`          | `int` or `list`          | `None`   | Freezes the first N layers of the model or specified layers by index, reducing the number of trainable parameters. Useful for fine-tuning or [transfer learning](https://www.ultralytics.com/glossary/transfer-learning).                                                               |
| `imgsz`           | `int`                    | `640`    | Target image size for training. Images are resized to squares with sides equal to the specified value (if `rect=False`), preserving aspect ratio for YOLO models but not RT-DETR. Affects model [accuracy](https://www.ultralytics.com/glossary/accuracy) and computational complexity. |
| `kobj`            | `float`                  | `2.0`    | Weight of the keypoint objectness loss in pose estimation models, balancing detection confidence with pose accuracy.                                                                                                                                                                    |
| `lr0`             | `float`                  | `0.01`   | Initial learning rate (i.e. `SGD=1E-2`, `Adam=1E-3`). Adjusting this value is crucial for the optimization process, influencing how rapidly model weights are updated.                                                                                                                  |
| `lrf`             | `float`                  | `0.01`   | Final learning rate as a fraction of the initial rate = (`lr0 * lrf`), used in conjunction with schedulers to adjust the learning rate over time.                                                                                                                                       |
| `mask_ratio`      | `int`                    | `4`      | Downsample ratio for segmentation masks, affecting the resolution of masks used during training.                                                                                                                                                                                        |
| `model`           | `str`                    | `None`   | Specifies the model file for training. Accepts a path to either a `.pt` pretrained model or a `.yaml` configuration file. Essential for defining the model structure or initializing weights.                                                                                           |
| `momentum`        | `float`                  | `0.937`  | Momentum factor for SGD or beta1 for [Adam optimizers](https://www.ultralytics.com/glossary/adam-optimizer), influencing the incorporation of past gradients in the current update.                                                                                                     |
| `multi_scale`     | `bool`                   | `False`  | Enables multi-scale training by increasing/decreasing `imgsz` by up to a factor of `0.5` during training. Trains the model to be more accurate with multiple `imgsz` during inference.                                                                                                  |
| `name`            | `str`                    | `None`   | Name of the training run. Used for creating a subdirectory within the project folder, where training logs and outputs are stored.                                                                                                                                                       |
| `nbs`             | `int`                    | `64`     | Nominal batch size for normalization of loss.                                                                                                                                                                                                                                           |
| `optimizer`       | `str`                    | `'auto'` | Choice of optimizer for training. Options include `SGD`, `Adam`, `AdamW`, `NAdam`, `RAdam`, `RMSProp` etc., or `auto` for automatic selection based on model configuration. Affects convergence speed and stability.                                                                    |
| `overlap_mask`    | `bool`                   | `True`   | Determines whether object masks should be merged into a single mask for training, or kept separate for each object. In case of overlap, the smaller mask is overlaid on top of the larger mask during merge.                                                                            |
| `patience`        | `int`                    | `100`    | Number of epochs to wait without improvement in validation metrics before early stopping the training. Helps prevent [overfitting](https://www.ultralytics.com/glossary/overfitting) by stopping training when performance plateaus.                                                    |
| `plots`           | `bool`                   | `False`  | Generates and saves plots of training and validation metrics, as well as prediction examples, providing visual insights into model performance and learning progression.                                                                                                                |
| `pose`            | `float`                  | `12.0`   | Weight of the pose loss in models trained for pose estimation, influencing the emphasis on accurately predicting pose keypoints.                                                                                                                                                        |
| `pretrained`      | `bool` or `str`          | `True`   | Determines whether to start training from a pretrained model. Can be a boolean value or a string path to a specific model from which to load weights. Enhances training efficiency and model performance.                                                                               |
| `profile`         | `bool`                   | `False`  | Enables profiling of ONNX and TensorRT speeds during training, useful for optimizing model deployment.                                                                                                                                                                                  |
| `project`         | `str`                    | `None`   | Name of the project directory where training outputs are saved. Allows for organized storage of different experiments.                                                                                                                                                                  |
| `rect`            | `bool`                   | `False`  | Enables minimum padding strategy—images in a batch are minimally padded to reach a common size, with the longest side equal to `imgsz`. Can improve efficiency and speed but may affect model accuracy.                                                                                 |
| `resume`          | `bool`                   | `False`  | Resumes training from the last saved checkpoint. Automatically loads model weights, optimizer state, and epoch count, continuing training seamlessly.                                                                                                                                   |
| `save`            | `bool`                   | `True`   | Enables saving of training checkpoints and final model weights. Useful for resuming training or [model deployment](https://www.ultralytics.com/glossary/model-deployment).                                                                                                              |
| `save_period`     | `int`                    | `-1`     | Frequency of saving model checkpoints, specified in epochs. A value of -1 disables this feature. Useful for saving interim models during long training sessions.                                                                                                                        |
| `seed`            | `int`                    | `0`      | Sets the random seed for training, ensuring reproducibility of results across runs with the same configurations.                                                                                                                                                                        |
| `single_cls`      | `bool`                   | `False`  | Treats all classes in multi-class datasets as a single class during training. Useful for binary classification tasks or when focusing on object presence rather than classification.                                                                                                    |
| `time`            | `float`                  | `None`   | Maximum training time in hours. If set, this overrides the `epochs` argument, allowing training to automatically stop after the specified duration. Useful for time-constrained training scenarios.                                                                                     |
| `val`             | `bool`                   | `True`   | Enables validation during training, allowing for periodic evaluation of model performance on a separate dataset.                                                                                                                                                                        |
| `warmup_bias_lr`  | `float`                  | `0.1`    | Learning rate for bias parameters during the warmup phase, helping stabilize model training in the initial epochs.                                                                                                                                                                      |
| `warmup_epochs`   | `float`                  | `3.0`    | Number of epochs for learning rate warmup, gradually increasing the learning rate from a low value to the initial learning rate to stabilize training early on.                                                                                                                         |
| `warmup_momentum` | `float`                  | `0.8`    | Initial momentum for warmup phase, gradually adjusting to the set momentum over the warmup period.                                                                                                                                                                                      |
| `weight_decay`    | `float`                  | `0.0005` | L2 [regularization](https://www.ultralytics.com/glossary/regularization) term, penalizing large weights to prevent overfitting.                                                                                                                                                         |
| `workers`         | `int`                    | `8`      | Number of worker threads for data loading (per `RANK` if Multi-GPU training). Influences the speed of data preprocessing and feeding into the model, especially useful in multi-GPU setups.                                                                                             |
### 2. Augmentation Settings and Hyperparameters
| Argument                                                                                                                       | Type    | Default       | Supported Tasks                                | Range         | Description                                                                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------ | ------- | ------------- | ---------------------------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`hsv_h`](https://docs.ultralytics.com/guides/yolo-data-augmentation//#hue-adjustment-hsv_h)                                   | `float` | `0.015`       | `detect`, `segment`, `pose`, `obb`, `classify` | `0.0 - 1.0`   | Adjusts the hue of the image by a fraction of the color wheel, introducing color variability. Helps the model generalize across different lighting conditions. |
| [`hsv_s`](https://docs.ultralytics.com/guides/yolo-data-augmentation//#saturation-adjustment-hsv_s)                            | `float` | `0.7`         | `detect`, `segment`, `pose`, `obb`, `classify` | `0.0 - 1.0`   | Alters the saturation of the image by a fraction, affecting the intensity of colors. Useful for simulating different environmental conditions.                 |
| [`hsv_v`](https://docs.ultralytics.com/guides/yolo-data-augmentation//#brightness-adjustment-hsv_v)                            | `float` | `0.4`         | `detect`, `segment`, `pose`, `obb`, `classify` | `0.0 - 1.0`   | Modifies the value (brightness) of the image by a fraction, helping the model to perform well under various lighting conditions.                               |
| [`degrees`](https://docs.ultralytics.com/guides/yolo-data-augmentation//#rotation-degrees)                                     | `float` | `0.0`         | `detect`, `segment`, `pose`, `obb`             | `0.0 - 180`   | Rotates the image randomly within the specified degree range, improving the model's ability to recognize objects at various orientations.                      |
| [`translate`](https://docs.ultralytics.com/guides/yolo-data-augmentation//#translation-translate)                              | `float` | `0.1`         | `detect`, `segment`, `pose`, `obb`             | `0.0 - 1.0`   | Translates the image horizontally and vertically by a fraction of the image size, aiding in learning to detect partially visible objects.                      |
| [`scale`](https://docs.ultralytics.com/guides/yolo-data-augmentation//#scale-scale)                                            | `float` | `0.5`         | `detect`, `segment`, `pose`, `obb`, `classify` | `>=0.0`       | Scales the image by a gain factor, simulating objects at different distances from the camera.                                                                  |
| [`shear`](https://docs.ultralytics.com/guides/yolo-data-augmentation//#shear-shear)                                            | `float` | `0.0`         | `detect`, `segment`, `pose`, `obb`             | `-180 - +180` | Shears the image by a specified degree, mimicking the effect of objects being viewed from different angles.                                                    |
| [`perspective`](https://docs.ultralytics.com/guides/yolo-data-augmentation//#perspective-perspective)                          | `float` | `0.0`         | `detect`, `segment`, `pose`, `obb`             | `0.0 - 0.001` | Applies a random perspective transformation to the image, enhancing the model's ability to understand objects in 3D space.                                     |
| [`flipud`](https://docs.ultralytics.com/guides/yolo-data-augmentation//#flip-up-down-flipud)                                   | `float` | `0.0`         | `detect`, `segment`, `pose`, `obb`, `classify` | `0.0 - 1.0`   | Flips the image upside down with the specified probability, increasing the data variability without affecting the object's characteristics.                    |
| [`fliplr`](https://docs.ultralytics.com/guides/yolo-data-augmentation//#flip-left-right-fliplr)                                | `float` | `0.5`         | `detect`, `segment`, `pose`, `obb`, `classify` | `0.0 - 1.0`   | Flips the image left to right with the specified probability, useful for learning symmetrical objects and increasing dataset diversity.                        |
| [`bgr`](https://docs.ultralytics.com/guides/yolo-data-augmentation//#bgr-channel-swap-bgr)                                     | `float` | `0.0`         | `detect`, `segment`, `pose`, `obb`             | `0.0 - 1.0`   | Flips the image channels from RGB to BGR with the specified probability, useful for increasing robustness to incorrect channel ordering.                       |
| [`mosaic`](https://docs.ultralytics.com/guides/yolo-data-augmentation//#mosaic-mosaic)                                         | `float` | `1.0`         | `detect`, `segment`, `pose`, `obb`             | `0.0 - 1.0`   | Combines four training images into one, simulating different scene compositions and object interactions. Highly effective for complex scene understanding.     |
| [`mixup`](https://docs.ultralytics.com/guides/yolo-data-augmentation//#mixup-mixup)                                            | `float` | `0.0`         | `detect`, `segment`, `pose`, `obb`             | `0.0 - 1.0`   | Blends two images and their labels, creating a composite image. Enhances the model's ability to generalize by introducing label noise and visual variability.  |
| [`cutmix`](https://docs.ultralytics.com/guides/yolo-data-augmentation//#cutmix-cutmix)                                         | `float` | `0.0`         | `detect`, `segment`, `pose`, `obb`             | `0.0 - 1.0`   | Combines portions of two images, creating a partial blend while maintaining distinct regions. Enhances model robustness by creating occlusion scenarios.       |
| [`copy_paste`](https://docs.ultralytics.com/guides/yolo-data-augmentation//#copy-paste-copy_paste)                             | `float` | `0.0`         | `segment`                                      | `0.0 - 1.0`   | Copies and pastes objects across images to increase object instances.                                                                                          |
| [`copy_paste_mode`](https://docs.ultralytics.com/guides/yolo-data-augmentation//#copy-paste-mode-copy_paste_mode)              | `str`   | `flip`        | `segment`                                      | -             | Specifies the `copy-paste` strategy to use. Options include `'flip'` and `'mixup'`.                                                                            |
| [`auto_augment`](https://docs.ultralytics.com/guides/yolo-data-augmentation//#auto-augment-auto_augment)                       | `str`   | `randaugment` | `classify`                                     | -             | Applies a predefined augmentation policy (`'randaugment'`, `'autoaugment'`, or `'augmix'`) to enhance model performance through visual diversity.              |
| [`erasing`](https://docs.ultralytics.com/guides/yolo-data-augmentation//#random-erasing-erasing)                               | `float` | `0.4`         | `classify`                                     | `0.0 - 0.9`   | Randomly erases regions of the image during training to encourage the model to focus on less obvious features.                                                 |
| [`augmentations`](https://docs.ultralytics.com/guides/yolo-data-augmentation//#custom-albumentations-transforms-augmentations) | `list`  | ``            | `detect`, `segment`, `pose`, `obb`             | -             | Custom Albumentations transforms for advanced data augmentation (Python API only). Accepts a list of transform objects for specialized augmentation needs.     |

##Inference
#### 1. Inference Sources

| Source                                                | Example                                    | Type            | Notes                                                                                       |
| ----------------------------------------------------- | ------------------------------------------ | --------------- | ------------------------------------------------------------------------------------------- |
| YouTube ✅                                             | `'https://youtu.be/LNwODJXcvt4'`           | `str`           | URL to a YouTube video.                                                                     |
| webcam ✅                                              | `0`                                        | `int`           | Index of the connected camera device to run inference on.                                   |
| video ✅                                               | `'video.mp4'`                              | `str` or `Path` | Video file in formats like MP4, AVI, etc.                                                   |
| URL                                                   | `'https://ultralytics.com/images/bus.jpg'` | `str`           | URL to an image.                                                                            |
| torch                                                 | `torch.zeros(16,3,320,640)`                | `torch.Tensor`  | BCHW format with RGB channels `float32 (0.0-1.0)`.                                          |
| stream ✅                                              | `'rtsp://example.com/media.mp4'`           | `str`           | URL for streaming protocols such as RTSP, RTMP, TCP, or an IP address.                      |
| screenshot                                            | `'screen'`                                 | `str`           | Capture a screenshot.                                                                       |
| PIL                                                   | `Image.open('image.jpg')`                  | `PIL.Image`     | HWC format with RGB channels.                                                               |
| [OpenCV](https://www.ultralytics.com/glossary/opencv) | `cv2.imread('image.jpg')`                  | `np.ndarray`    | HWC format with BGR channels `uint8 (0-255)`.                                               |
| numpy                                                 | `np.zeros((640,1280,3))`                   | `np.ndarray`    | HWC format with BGR channels `uint8 (0-255)`.                                               |
| multi-stream ✅                                        | `'list.streams'`                           | `str` or `Path` | `*.streams` text file with one stream URL per row, i.e. 8 streams will run at batch-size 8. |
| image                                                 | `'image.jpg'`                              | `str` or `Path` | Single image file.                                                                          |
| glob ✅                                                | `'path/*.jpg'`                             | `str`           | Glob pattern to match multiple files. Use the `*` character as a wildcard.                  |
| directory ✅                                           | `'path/'`                                  | `str` or `Path` | Path to a directory containing images or videos.                                            |
| CSV                                                   | `'sources.csv'`                            | `str` or `Path` | CSV file containing paths to images, videos, or directories.                                |

#### 2. Inference arguments
|Argument|Type|Default|Description|
|---|---|---|---|
|`agnostic_nms`|`bool`|`False`|Enables class-agnostic Non-Maximum Suppression (NMS), which merges overlapping boxes of different classes. Useful in multi-class detection scenarios where class overlap is common.|
|`augment`|`bool`|`False`|Enables test-time augmentation (TTA) for predictions, potentially improving detection robustness at the cost of inference speed.|
|`batch`|`int`|`1`|Specifies the batch size for inference (only works when the source is [a directory, video file, or `.txt` file](https://docs.ultralytics.com/modes/predict/#inference-sources)). A larger batch size can provide higher throughput, shortening the total amount of time required for inference.|
|`classes`|`list[int]`|`None`|Filters predictions to a set of class IDs. Only detections belonging to the specified classes will be returned. Useful for focusing on relevant objects in multi-class detection tasks.|
|`compile`|`bool` or `str`|`False`|Enables PyTorch 2.x `torch.compile` graph compilation with `backend='inductor'`. Accepts `True` → `"default"`, `False` → disables, or a string mode such as `"default"`, `"reduce-overhead"`, `"max-autotune-no-cudagraphs"`. Falls back to eager with a warning if unsupported.|
|`conf`|`float`|`0.25`|Sets the minimum confidence threshold for detections. Objects detected with confidence below this threshold will be disregarded. Adjusting this value can help reduce false positives.|
|`device`|`str`|`None`|Specifies the device for inference (e.g., `cpu`, `cuda:0` or `0`). Allows users to select between CPU, a specific GPU, or other compute devices for model execution.|
|`embed`|`list[int]`|`None`|Specifies the layers from which to extract feature vectors or [embeddings](https://www.ultralytics.com/glossary/embeddings). Useful for downstream tasks like clustering or similarity search.|
|`half`|`bool`|`False`|Enables half-[precision](https://www.ultralytics.com/glossary/precision) (FP16) inference, which can speed up model inference on supported GPUs with minimal impact on accuracy.|
|`imgsz`|`int` or `tuple`|`640`|Defines the image size for inference. Can be a single integer `640` for square resizing or a (height, width) tuple. Proper sizing can improve detection [accuracy](https://www.ultralytics.com/glossary/accuracy) and processing speed.|
|`iou`|`float`|`0.7`|[Intersection Over Union](https://www.ultralytics.com/glossary/intersection-over-union-iou) (IoU) threshold for Non-Maximum Suppression (NMS). Lower values result in fewer detections by eliminating overlapping boxes, useful for reducing duplicates.|
|`max_det`|`int`|`300`|Maximum number of detections allowed per image. Limits the total number of objects the model can detect in a single inference, preventing excessive outputs in dense scenes.|
|`name`|`str`|`None`|Name of the prediction run. Used for creating a subdirectory within the project folder, where prediction outputs are stored if `save` is enabled.|
|`project`|`str`|`None`|Name of the project directory where prediction outputs are saved if `save` is enabled.|
|`rect`|`bool`|`True`|If enabled, minimally pads the shorter side of the image until it's divisible by stride to improve inference speed. If disabled, pads the image to a square during inference.|
|`retina_masks`|`bool`|`False`|Returns high-resolution segmentation masks. The returned masks (`masks.data`) will match the original image size if enabled. If disabled, they have the image size used during inference.|
|`source`|`str`|`'ultralytics/assets'`|Specifies the data source for inference. Can be an image path, video file, directory, URL, or device ID for live feeds. Supports a wide range of formats and sources, enabling flexible application across [different types of input](https://docs.ultralytics.com/modes/predict/#inference-sources).|
|`stream`|`bool`|`False`|Enables memory-efficient processing for long videos or numerous images by returning a generator of Results objects instead of loading all frames into memory at once.|
|`stream_buffer`|`bool`|`False`|Determines whether to queue incoming frames for video streams. If `False`, old frames get dropped to accommodate new frames (optimized for real-time applications). If `True`, queues new frames in a buffer, ensuring no frames get skipped, but will cause latency if inference FPS is lower than stream FPS.|
|`verbose`|`bool`|`True`|Controls whether to display detailed inference logs in the terminal, providing real-time feedback on the prediction process.|
|`vid_stride`|`int`|`1`|Frame stride for video inputs. Allows skipping frames in videos to speed up processing at the cost of temporal resolution. A value of 1 processes every frame, higher values skip frames.|
|`visualize`|`bool`|`False`|Activates visualization of model features during inference, providing insights into what the model is "seeing". Useful for debugging and model interpretation.|
#### 3. Visualization arguments
|Argument|Type|Default|Description|
|---|---|---|---|
|`line_width`|`None or int`|`None`|Specifies the line width of bounding boxes. If `None`, the line width is automatically adjusted based on the image size. Provides visual customization for clarity.|
|`save`|`bool`|`False or True`|Enables saving of the annotated images or videos to files. Useful for documentation, further analysis, or sharing results. Defaults to True when using CLI & False when used in Python.|
|`save_conf`|`bool`|`False`|Includes confidence scores in the saved text files. Enhances the detail available for post-processing and analysis.|
|`save_crop`|`bool`|`False`|Saves cropped images of detections. Useful for dataset augmentation, analysis, or creating focused datasets for specific objects.|
|`save_frames`|`bool`|`False`|When processing videos, saves individual frames as images. Useful for extracting specific frames or for detailed frame-by-frame analysis.|
|`save_txt`|`bool`|`False`|Saves detection results in a text file, following the format `[class] [x_center] [y_center] [width] [height] [confidence]`. Useful for integration with other analysis tools.|
|`show`|`bool`|`False`|If `True`, displays the annotated images or videos in a window. Useful for immediate visual feedback during development or testing.|
|`show_boxes`|`bool`|`True`|Draws bounding boxes around detected objects. Essential for visual identification and location of objects in images or video frames.|
|`show_conf`|`bool`|`True`|Displays the confidence score for each detection alongside the label. Gives insight into the model's certainty for each detection.|
|`show_labels`|`bool`|`True`|Displays labels for each detection in the visual output. Provides immediate understanding of detected objects.|
#### 4. Image and Video Formats
1. Images

|Image Suffixes|Example Predict Command|Reference|
|---|---|---|
|`.bmp`|`yolo predict source=image.bmp`|[Microsoft BMP File Format](https://en.wikipedia.org/wiki/BMP_file_format)|
|`.dng`|`yolo predict source=image.dng`|[Adobe DNG](https://en.wikipedia.org/wiki/Digital_Negative)|
|`.HEIC`|`yolo predict source=image.HEIC`|[High Efficiency Image Format](https://en.wikipedia.org/wiki/HEIF)|
|`.jpeg`|`yolo predict source=image.jpeg`|[JPEG](https://en.wikipedia.org/wiki/JPEG)|
|`.jpg`|`yolo predict source=image.jpg`|[JPEG](https://en.wikipedia.org/wiki/JPEG)|
|`.mpo`|`yolo predict source=image.mpo`|[Multi Picture Object](https://fileinfo.com/extension/mpo)|
|`.pfm`|`yolo predict source=image.pfm`|[Portable FloatMap](https://en.wikipedia.org/wiki/Netpbm#File_formats)|
|`.png`|`yolo predict source=image.png`|[Portable Network Graphics](https://en.wikipedia.org/wiki/PNG)|
|`.tif`|`yolo predict source=image.tif`|[Tag Image File Format](https://en.wikipedia.org/wiki/TIFF)|
|`.tiff`|`yolo predict source=image.tiff`|[Tag Image File Format](https://en.wikipedia.org/wiki/TIFF)|
|`.webp`|`yolo predict source=image.webp`|[WebP](https://en.wikipedia.org/wiki/WebP)|
2. Videos

|Video Suffixes|Example Predict Command|Reference|
|---|---|---|
|`.wmv`|`yolo predict source=video.wmv`|[Windows Media Video](https://en.wikipedia.org/wiki/Windows_Media_Video)|
|`.webm`|`yolo predict source=video.webm`|[WebM Project](https://en.wikipedia.org/wiki/WebM)|
|`.ts`|`yolo predict source=video.ts`|[MPEG Transport Stream](https://en.wikipedia.org/wiki/MPEG_transport_stream)|
|`.mpg`|`yolo predict source=video.mpg`|[MPEG-1 Part 2](https://en.wikipedia.org/wiki/MPEG-1)|
|`.mpeg`|`yolo predict source=video.mpeg`|[MPEG-1 Part 2](https://en.wikipedia.org/wiki/MPEG-1)|
|`.mp4`|`yolo predict source=video.mp4`|[MPEG-4 Part 14 - Wikipedia](https://en.wikipedia.org/wiki/MPEG-4_Part_14)|
|`.mov`|`yolo predict source=video.mov`|[QuickTime File Format](https://en.wikipedia.org/wiki/QuickTime_File_Format)|
|`.mkv`|`yolo predict source=video.mkv`|[Matroska](https://en.wikipedia.org/wiki/Matroska)|
|`.m4v`|`yolo predict source=video.m4v`|[MPEG-4 Part 14](https://en.wikipedia.org/wiki/M4V)|
|`.gif`|`yolo predict source=video.gif`|[Graphics Interchange Format](https://en.wikipedia.org/wiki/GIF)|
|`.avi`|`yolo predict source=video.avi`|[Audio Video Interleave](https://en.wikipedia.org/wiki/Audio_Video_Interleave)|
|`.asf`|`yolo predict source=video.asf`|[Advanced Systems Format](https://en.wikipedia.org/wiki/Advanced_Systems_Format)|

### 5. Results
`Results` objects have the following attributes:

|Attribute|Type|Description|
|---|---|---|
|`orig_img`|`np.ndarray`|The original image as a numpy array.|
|`orig_shape`|`tuple`|The original image shape in (height, width) format.|
|`boxes`|`Boxes, optional`|A Boxes object containing the detection bounding boxes.|
|`masks`|`Masks, optional`|A Masks object containing the detection masks.|
|`probs`|`Probs, optional`|A Probs object containing probabilities of each class for classification task.|
|`keypoints`|`Keypoints, optional`|A Keypoints object containing detected keypoints for each object.|
|`obb`|`OBB, optional`|An OBB object containing oriented bounding boxes.|
|`speed`|`dict`|A dictionary of preprocess, inference, and postprocess speeds in milliseconds per image.|
|`names`|`dict`|A dictionary mapping class indices to class names.|
|`path`|`str`|The path to the image file.|
|`save_dir`|`str, optional`|Directory to save results.|

`Results` objects have the following methods:

|Method|Return Type|Description|
|---|---|---|
|`update()`|`None`|Updates the Results object with new detection data (boxes, masks, probs, obb, keypoints).|
|`cpu()`|`Results`|Returns a copy of the Results object with all tensors moved to CPU memory.|
|`numpy()`|`Results`|Returns a copy of the Results object with all tensors converted to numpy arrays.|
|`cuda()`|`Results`|Returns a copy of the Results object with all tensors moved to GPU memory.|
|`to()`|`Results`|Returns a copy of the Results object with tensors moved to specified device and dtype.|
|`new()`|`Results`|Creates a new Results object with the same image, path, names, and speed attributes.|
|`plot()`|`np.ndarray`|Plots detection results on an input RGB image and returns the annotated image.|
|`show()`|`None`|Displays the image with annotated inference results.|
|`save()`|`str`|Saves annotated inference results image to file and returns the filename.|
|`verbose()`|`str`|Returns a log string for each task, detailing detection and classification outcomes.|
|`save_txt()`|`str`|Saves detection results to a text file and returns the path to the saved file.|
|`save_crop()`|`None`|Saves cropped detection images to specified directory.|
|`summary()`|`List[Dict[str, Any]]`|Converts inference results to a summarized dictionary with optional normalization.|
|`to_df()`|`DataFrame`|Converts detection results to a Polars DataFrame.|
|`to_csv()`|`str`|Converts detection results to CSV format.|
|`to_json()`|`str`|Converts detection results to JSON format.|

#### 6. Boxes
|Name|Type|Description|
|---|---|---|
|`cls`|Property (`torch.Tensor`)|Return the class values of the boxes.|
|`conf`|Property (`torch.Tensor`)|Return the confidence values of the boxes.|
|`cpu()`|Method|Move the object to CPU memory.|
|`cuda()`|Method|Move the object to CUDA memory.|
|`id`|Property (`torch.Tensor`)|Return the track IDs of the boxes (if available).|
|`numpy()`|Method|Convert the object to a numpy array.|
|`to()`|Method|Move the object to the specified device.|
|`xywh`|Property (`torch.Tensor`)|Return the boxes in xywh format.|
|`xywhn`|Property (`torch.Tensor`)|Return the boxes in xywh format normalized by original image size.|
|`xyxy`|Property (`torch.Tensor`)|Return the boxes in xyxy format.|
|`xyxyn`|Property (`torch.Tensor`)|Return the boxes in xyxy format normalized by original image size.|
#### 7. Masks
|Name|Type|Description|
|---|---|---|
|`cpu()`|Method|Returns the masks tensor on CPU memory.|
|`cuda()`|Method|Returns the masks tensor on GPU memory.|
|`numpy()`|Method|Returns the masks tensor as a numpy array.|
|`to()`|Method|Returns the masks tensor with the specified device and dtype.|
|`xy`|Property (`torch.Tensor`)|A list of segments in pixel coordinates represented as tensors.|
|`xyn`|Property (`torch.Tensor`)|A list of normalized segments represented as tensors.|
#### 8. Keypoints
| Name      | Type                      | Description                                                       |
| --------- | ------------------------- | ----------------------------------------------------------------- |
| `conf`    | Property (`torch.Tensor`) | Returns confidence values of keypoints if available, else None.   |
| `cpu()`   | Method                    | Returns the keypoints tensor on CPU memory.                       |
| `cuda()`  | Method                    | Returns the keypoints tensor on GPU memory.                       |
| `numpy()` | Method                    | Returns the keypoints tensor as a numpy array.                    |
| `to()`    | Method                    | Returns the keypoints tensor with the specified device and dtype. |
| `xy`      | Property (`torch.Tensor`) | A list of keypoints in pixel coordinates represented as tensors.  |
| `xyn`     | Property (`torch.Tensor`) | A list of normalized keypoints represented as tensors.            |
#### 9. Probs
|Name|Type|Description|
|---|---|---|
|`cpu()`|Method|Returns a copy of the probs tensor on CPU memory.|
|`cuda()`|Method|Returns a copy of the probs tensor on GPU memory.|
|`numpy()`|Method|Returns a copy of the probs tensor as a numpy array.|
|`to()`|Method|Returns a copy of the probs tensor with the specified device and dtype.|
|`top1`|Property (`int`)|Index of the top 1 class.|
|`top1conf`|Property (`torch.Tensor`)|Confidence of the top 1 class.|
|`top5`|Property (`list[int]`)|Indices of the top 5 classes.|
|`top5conf`|Property (`torch.Tensor`)|Confidences of the top 5 classes.|
#### 10. OBB
|Name|Type|Description|
|---|---|---|
|`cls`|Property (`torch.Tensor`)|Return the class values of the boxes.|
|`conf`|Property (`torch.Tensor`)|Return the confidence values of the boxes.|
|`cpu()`|Method|Move the object to CPU memory.|
|`cuda()`|Method|Move the object to CUDA memory.|
|`id`|Property (`torch.Tensor`)|Return the track IDs of the boxes (if available).|
|`numpy()`|Method|Convert the object to a numpy array.|
|`to()`|Method|Move the object to the specified device.|
|`xywhr`|Property (`torch.Tensor`)|Return the rotated boxes in xywhr format.|
|`xyxy`|Property (`torch.Tensor`)|Return the horizontal boxes in xyxy format.|
|`xyxyxyxy`|Property (`torch.Tensor`)|Return the rotated boxes in xyxyxyxy format.|
|`xyxyxyxyn`|Property (`torch.Tensor`)|Return the rotated boxes in xyxyxyxy format normalized by image size.|
#### 11.Plotting Results
`plot()` Method Parameters

|Argument|Type|Description|Default|
|---|---|---|---|
|`boxes`|`bool`|Overlay bounding boxes on the image.|`True`|
|`color_mode`|`str`|Specify the color mode, e.g., 'instance' or 'class'.|`'class'`|
|`conf`|`bool`|Include detection confidence scores.|`True`|
|`filename`|`str`|Path and name of the file to save the annotated image if `save` is `True`.|`None`|
|`font`|`str`|Font name for text annotations.|`'Arial.ttf'`|
|`font_size`|`float`|Text font size. Scales with image size if `None`.|`None`|
|`im_gpu`|`torch.Tensor`|GPU-accelerated image for faster mask plotting. Shape: (1, 3, 640, 640).|`None`|
|`img`|`np.ndarray`|Alternative image for plotting. Uses the original image if `None`.|`None`|
|`kpt_line`|`bool`|Connect keypoints with lines.|`True`|
|`kpt_radius`|`int`|Radius for drawn keypoints.|`5`|
|`labels`|`bool`|Include class labels in annotations.|`True`|
|`line_width`|`float`|Line width of bounding boxes. Scales with image size if `None`.|`None`|
|`masks`|`bool`|Overlay masks on the image.|`True`|
|`pil`|`bool`|Return image as a PIL Image object.|`False`|
|`probs`|`bool`|Include classification probabilities.|`True`|
|`save`|`bool`|Save the annotated image to a file specified by `filename`.|`False`|
|`show`|`bool`|Display the annotated image directly using the default image viewer.|`False`|
|`txt_color`|`tuple[int, int, int]`|RGB text color for bounding box and image classification label.|`(255, 255, 255)`|