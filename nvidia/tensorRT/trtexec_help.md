

TensorRT 官方工具 trtexec 的完整指令參數說明

🧠 TensorRT trtexec 參數說明（繁體中文完整翻譯）

=== 模型選項 (Model Options) ===

| 參數              | 說明                |
| --------------- | ----------------- |
| `--onnx=<file>` | 指定要載入的 ONNX 模型檔案。 |

=== 建置選項 (Build Options) ===

| 參數                                                         | 說明                                                                                 |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `--minShapes=spec`                                         | 設定動態 shape 的最小尺寸。                                                                  |
| `--optShapes=spec`                                         | 設定動態 shape 的最佳尺寸（最佳化使用）。                                                           |
| `--maxShapes=spec`                                         | 設定動態 shape 的最大尺寸。                                                                  |
| `--minShapesCalib`, `--optShapesCalib`, `--maxShapesCalib` | 在 INT8 校正 (calibration) 階段使用的動態 shape 設定。<br>※ 若只給 `optShapes`，會自動將 min/max 設成相同值。 |
| **範例：**                                                    | `input0:1x3x256x256,input1:1x3x128x128`                                            |
| **特殊情況：**                                                  | 0-D（scalar）可用 `input0:scalar` 或 `input0:` 表示。                                      |

I/O 格式

| 參數                       | 說明                                                                                    |
| ------------------------ | ------------------------------------------------------------------------------------- |
| `--inputIOFormats=spec`  | 指定輸入 tensor 的資料型態與格式。<br>預設：`fp32:chw`。<br>若要指定多個輸入，必須依據輸入順序全部設定。                     |
| `--outputIOFormats=spec` | 指定輸出 tensor 的資料型態與格式。<br>格式語法：`type:fmt`，例如 `fp16:chw4`。                              |
| 可用格式                     | `chw`, `chw2`, `chw4`, `chw16`, `chw32`, `hwc8`, `hwc`, `dla_linear`, `dla_hwc4`, ... |

記憶體設定

| 參數                       | 說明                                                                 |
| ------------------------ | ------------------------------------------------------------------ |
| `--memPoolSize=poolspec` | 限制特定記憶體池的大小，例如：`workspace:2048M`。<br>單位可用 B, K, M, G；不加單位則預設為 MiB。 |

建構控制

| 參數                             | 說明                                                         |
| ------------------------------ | ---------------------------------------------------------- |
| `--profilingVerbosity=mode`    | 設定 profiler 輸出層級：`layer_names_only` / `detailed` / `none`。 |
| `--avgTiming=M`                | 每次 kernel 選擇平均測試次數（預設 8 次）。                                |
| `--refit`                      | 允許 engine 支援 refit（重設 layer 權重）。                           |
| `--stripWeights`               | 移除權重資料（常搭配 refit 使用）。                                      |
| `--stripAllWeights`            | 移除所有可 refit 權重。                                            |
| `--versionCompatible` 或 `--vc` | 讓 engine 能在新版本 TensorRT 上使用。                               |
| `--pluginInstanceNorm`, `--pi` | 強制使用 plugin 實作 InstanceNorm，而非原生版本。                        |
| `--useRuntime=runtime`         | 指定執行 runtime 類型：`full`、`lean` 或 `dispatch`。                |
| `--leanDLLPath=<file>`         | 指定 lean runtime 外部 DLL。                                    |
| `--excludeLeanRuntime`         | 在啟用 `--versionCompatible` 時，排除內建的 lean runtime。            |

精度與最佳化

| 參數                                                        | 說明                                          |
| --------------------------------------------------------- | ------------------------------------------- |
| `--sparsity=spec`                                         | 控制稀疏化策略：`disable` / `enable` / `force`。     |
| `--noTF32`                                                | 關閉 TF32 精度（預設開啟）。                           |
| `--fp16`, `--bf16`, `--int8`, `--fp8`, `--int4`, `--best` | 啟用不同精度模式。`--best` 會嘗試所有精度以求最大效能。            |
| `--stronglyTyped`                                         | 建立強型別 network。                              |
| `--directIO`                                              | 避免在網路邊界進行資料 reformat。                       |
| `--precisionConstraints=spec`                             | 設定精度約束：`none` / `prefer` / `obey`。          |
| `--layerPrecisions=spec`                                  | 為特定 layer 指定精度（搭配 precisionConstraints 使用）。 |
| `--layerOutputTypes=spec`                                 | 為特定 layer 指定輸出資料型態。                         |
| `--layerDeviceTypes=spec`                                 | 指定某些 layer 使用 GPU 或 DLA。                    |
| `--calib=<file>`                                          | 讀取 INT8 校正 cache 檔案。                        |

安全與相容性

| 參數                                  | 說明                                       |
| ----------------------------------- | ---------------------------------------- |
| `--safe`                            | 建立 Safety 認證引擎（DLA 模式會自動開啟）。             |
| `--buildDLAStandalone`              | 生成可單獨由 cuDLA 載入的引擎。                      |
| `--allowGPUFallback`                | 若 DLA 不支援該層，可退回 GPU 執行。                  |
| `--restricted`                      | 啟用安全範圍檢查 (kSAFETY_SCOPE)。                |
| `--saveEngine=<file>`               | 儲存序列化後的 engine。                          |
| `--loadEngine=<file>`               | 載入序列化的 engine。                           |
| `--getPlanVersionOnly`              | 僅輸出 engine 的 TensorRT 版本，不進行反序列化。        |
| `--tacticSources=tactics`           | 指定要啟用或停用的 tactic 來源（如 +CUBLAS、-CUDNN）。   |
| `--noBuilderCache`                  | 關閉 builder 的 timing cache。               |
| `--timingCacheFile=<file>`          | 指定 timing cache 檔案位置。                    |
| `--builderOptimizationLevel=N`      | 設定建構最佳化等級（0~5，預設 3）。                     |
| `--hardwareCompatibilityLevel=mode` | 指定硬體相容層級：`none` 或 `ampere+`。             |
| `--runtimePlatform=platform`        | 設定目標執行平台：`SameAsBuild` 或 `WindowsAMD64`。 |

=== 推論選項 (Inference Options) ===

| 參數                          | 說明                                        |
| --------------------------- | ----------------------------------------- |
| `--shapes=spec`             | 設定推論時的輸入 shape。                           |
| `--loadInputs=spec`         | 從檔案載入輸入資料（預設為隨機產生）。                       |
| `--iterations=N`            | 執行至少 N 次推論（預設 10）。                        |
| `--warmUp=N`                | 預熱 N 毫秒（預設 200）。                          |
| `--duration=N`              | 測試時間至少 N 秒（預設 3）。                         |
| `--sleepTime=N`             | 推論啟動前延遲 N 毫秒。                             |
| `--infStreams=N`            | 建立 N 個執行 context 同時推論。                    |
| `--exposeDMA`               | 序列化 DMA 傳輸。                               |
| `--noDataTransfers`         | 停用 host/device 資料傳輸。                      |
| `--useManagedMemory`        | 使用 Unified Memory（統一記憶體）。                 |
| `--threads`                 | 啟用多執行緒執行推論。                               |
| `--useCudaGraph`            | 使用 CUDA Graph 捕獲與執行推論。                    |
| `--skipInference`           | 僅建立引擎，不執行推論。                              |
| `--allocationStrategy=spec` | 記憶體配置策略：`static` / `profile` / `runtime`。 |

=== 報告與輸出選項 (Reporting Options) ===

| 參數                          | 說明                          |
| --------------------------- | --------------------------- |
| `--verbose`                 | 顯示詳細 log。                   |
| `--avgRuns=N`               | 以 N 次迭代的平均值報告效能（預設 10）。     |
| `--percentile=P1,P2,...`    | 報告指定百分比的效能統計（預設 90,95,99%）。 |
| `--dumpOutput`              | 輸出最後一次推論結果。                 |
| `--dumpProfile`             | 顯示每層的時間分析。                  |
| `--dumpLayerInfo`           | 顯示每個 layer 的資訊。             |
| `--dumpOptimizationProfile` | 顯示最佳化 profile 的資訊。          |
| `--exportTimes=<file>`      | 將時間統計輸出成 JSON。              |
| `--exportOutput=<file>`     | 將輸出 tensor 寫入 JSON。         |
| `--exportProfile=<file>`    | 將每層的 profiler 資訊輸出 JSON。    |

=== 系統選項 (System Options) ===

| 參數                         | 說明                              |
| -------------------------- | ------------------------------- |
| `--device=N`               | 指定 CUDA 裝置 ID（預設 0）。            |
| `--useDLACore=N`           | 指定 DLA core ID。                 |
| `--staticPlugins`          | 靜態載入 plugin（可重複指定多次）。           |
| `--dynamicPlugins`         | 動態載入 plugin，並可序列化進 engine。      |
| `--setPluginsToSerialize`  | 指定要序列化進 engine 的 plugin。        |
| `--ignoreParsedPluginLibs` | 忽略 parser 自動包含的 plugin library。 |

=== 幫助 (Help) ===

| 參數             | 說明      |
| -------------- | ------- |
| `--help`, `-h` | 顯示完整說明。 |


``` bash
&&&& RUNNING TensorRT.trtexec [TensorRT v100500] [b18] # trtexec --help
=== Model Options ===
  --onnx=<file>               ONNX model

=== Build Options ===
  --minShapes=spec                   Build with dynamic shapes using a profile with the min shapes provided
  --optShapes=spec                   Build with dynamic shapes using a profile with the opt shapes provided
  --maxShapes=spec                   Build with dynamic shapes using a profile with the max shapes provided
  --minShapesCalib=spec              Calibrate with dynamic shapes using a profile with the min shapes provided
  --optShapesCalib=spec              Calibrate with dynamic shapes using a profile with the opt shapes provided
  --maxShapesCalib=spec              Calibrate with dynamic shapes using a profile with the max shapes provided
                                     Note: All three of min, opt and max shapes must be supplied.
                                           However, if only opt shapes is supplied then it will be expanded so
                                           that min shapes and max shapes are set to the same values as opt shapes.
                                           Input names can be wrapped with escaped single quotes (ex: 'Input:0').
                                     Example input shapes spec: input0:1x3x256x256,input1:1x3x128x128
                                     For scalars (0-D shapes), use input0:scalar or simply input0: with nothing after the colon.
                                     Each input shape is supplied as a key-value pair where key is the input name and
                                     value is the dimensions (including the batch dimension) to be used for that input.
                                     Each key-value pair has the key and value separated using a colon (:).
                                     Multiple input shapes can be provided via comma-separated key-value pairs, and each input name can
                                     contain at most one wildcard ('*') character.
  --inputIOFormats=spec              Type and format of each of the input tensors (default = all inputs in fp32:chw)
                                     See --outputIOFormats help for the grammar of type and format list.
                                     Note: If this option is specified, please set comma-separated types and formats for all
                                           inputs following the same order as network inputs ID (even if only one input
                                           needs specifying IO format) or set the type and format once for broadcasting.
  --outputIOFormats=spec             Type and format of each of the output tensors (default = all outputs in fp32:chw)
                                     Note: If this option is specified, please set comma-separated types and formats for all
                                           outputs following the same order as network outputs ID (even if only one output
                                           needs specifying IO format) or set the type and format once for broadcasting.
                                     IO Formats: spec  ::= IOfmt[","spec]
                                                 IOfmt ::= type:fmt
                                               type  ::= "fp32"|"fp16"|"bf16"|"int32"|"int64"|"int8"|"uint8"|"bool"
                                               fmt   ::= ("chw"|"chw2"|"chw4"|"hwc8"|"chw16"|"chw32"|"dhwc8"|
                                                          "cdhw32"|"hwc"|"dla_linear"|"dla_hwc4")["+"fmt]
  --memPoolSize=poolspec             Specify the size constraints of the designated memory pool(s)
                                     Supports the following base-2 suffixes: B (Bytes), G (Gibibytes), K (Kibibytes), M (Mebibytes).
                                     If none of suffixes is appended, the defualt unit is in MiB.
                                     Note: Also accepts decimal sizes, e.g. 0.25M. Will be rounded down to the nearest integer bytes.
                                     In particular, for dlaSRAM the bytes will be rounded down to the nearest power of 2.
                                   Pool constraint: poolspec ::= poolfmt[","poolspec]
                                                      poolfmt ::= pool:size
                                                    pool ::= "workspace"|"dlaSRAM"|"dlaLocalDRAM"|"dlaGlobalDRAM"|"tacticSharedMem"
  --profilingVerbosity=mode          Specify profiling verbosity. mode ::= layer_names_only|detailed|none (default = layer_names_only).
                                     Please only assign once.
  --avgTiming=M                      Set the number of times averaged in each iteration for kernel selection (default = 8)
  --refit                            Mark the engine as refittable. This will allow the inspection of refittable layers 
                                     and weights within the engine.
  --stripWeights                     Strip weights from plan. This flag works with either refit or refit with identical weights. Default
                                     to latter, but you can switch to the former by enabling both --stripWeights and --refit at the same
                                     time.
  --stripAllWeights                  Alias for combining the --refit and --stripWeights options. It marks all weights as refittable,
                                     disregarding any performance impact. Additionally, it strips all refittable weights after the 
                                     engine is built.
  --weightless                       [Deprecated] this knob has been deprecated. Please use --stripWeights
  --versionCompatible, --vc          Mark the engine as version compatible. This allows the engine to be used with newer versions
                                     of TensorRT on the same host OS, as well as TensorRTs dispatch and lean runtimes.
  --pluginInstanceNorm, --pi         Set `kNATIVE_INSTANCENORM` to false in the ONNX parser. This will cause the ONNX parser to use
                                     a plugin InstanceNorm implementation over the native implementation when parsing.
  --useRuntime=runtime               TensorRT runtime to execute engine. "lean" and "dispatch" require loading VC engine and do
                                     not support building an engine.
                                           runtime::= "full"|"lean"|"dispatch"
  --leanDLLPath=<file>               External lean runtime DLL to use in version compatiable mode.
  --excludeLeanRuntime               When --versionCompatible is enabled, this flag indicates that the generated engine should
                                     not include an embedded lean runtime. If this is set, the user must explicitly specify a
                                     valid lean runtime to use when loading the engine.
  --sparsity=spec                    Control sparsity (default = disabled). 
                                   Sparsity: spec ::= "disable", "enable", "force"
                                     Note: Description about each of these options is as below
                                           disable = do not enable sparse tactics in the builder (this is the default)
                                           enable  = enable sparse tactics in the builder (but these tactics will only be
                                                     considered if the weights have the right sparsity pattern)
                                           force   = enable sparse tactics in the builder and force-overwrite the weights to have
                                                     a sparsity pattern (even if you loaded a model yourself)
                                                     [Deprecated] this knob has been deprecated.
                                                     Please use <polygraphy surgeon prune> to rewrite the weights.
  --noTF32                           Disable tf32 precision (default is to enable tf32, in addition to fp32)
  --fp16                             Enable fp16 precision, in addition to fp32 (default = disabled)
  --bf16                             Enable bf16 precision, in addition to fp32 (default = disabled)
  --int8                             Enable int8 precision, in addition to fp32 (default = disabled)
  --fp8                              Enable fp8 precision, in addition to fp32 (default = disabled)
  --int4                             Enable int4 precision, in addition to fp32 (default = disabled)
  --best                             Enable all precisions to achieve the best performance (default = disabled)
  --stronglyTyped                    Create a strongly typed network. (default = disabled)
  --directIO                         Avoid reformatting at network boundaries. (default = disabled)
  --precisionConstraints=spec        Control precision constraint setting. (default = none)
                                       Precision Constraints: spec ::= "none" | "obey" | "prefer"
                                         none = no constraints
                                         prefer = meet precision constraints set by --layerPrecisions/--layerOutputTypes if possible
                                         obey = meet precision constraints set by --layerPrecisions/--layerOutputTypes or fail
                                                otherwise
  --layerPrecisions=spec             Control per-layer precision constraints. Effective only when precisionConstraints is set to
                                   "obey" or "prefer". (default = none)
                                   The specs are read left-to-right, and later ones override earlier ones. Each layer name can
                                     contain at most one wildcard ('*') character.
                                   Per-layer precision spec ::= layerPrecision[","spec]
                                                       layerPrecision ::= layerName":"precision
                                                       precision ::= "fp32"|"fp16"|"bf16"|"int32"|"int8"
  --layerOutputTypes=spec            Control per-layer output type constraints. Effective only when precisionConstraints is set to
                                   "obey" or "prefer". (default = none)
                                   The specs are read left-to-right, and later ones override earlier ones. Each layer name can
                                     contain at most one wildcard ('*') character. If a layer has more than
                                   one output, then multiple types separated by "+" can be provided for this layer.
                                   Per-layer output type spec ::= layerOutputTypes[","spec]
                                                         layerOutputTypes ::= layerName":"type
                                                         type ::= "fp32"|"fp16"|"bf16"|"int32"|"int8"["+"type]
  --layerDeviceTypes=spec            Specify layer-specific device type.
                                     The specs are read left-to-right, and later ones override earlier ones. If a layer does not have
                                     a device type specified, the layer will opt for the default device type.
                                   Per-layer device type spec ::= layerDeviceTypePair[","spec]
                                                         layerDeviceTypePair ::= layerName":"deviceType
                                                           deviceType ::= "GPU"|"DLA"
  --calib=<file>                     Read INT8 calibration cache file
  --safe                             Enable build safety certified engine, if DLA is enable, --buildDLAStandalone will be specified
                                     automatically (default = disabled)
  --buildDLAStandalone               Enable build DLA standalone loadable which can be loaded by cuDLA, when this option is enabled, 
                                     --allowGPUFallback is disallowed and --skipInference is enabled by default. Additionally, 
                                     specifying --inputIOFormats and --outputIOFormats restricts I/O data type and memory layout
                                     (default = disabled)
  --allowGPUFallback                 When DLA is enabled, allow GPU fallback for unsupported layers (default = disabled)
  --restricted                       Enable safety scope checking with kSAFETY_SCOPE build flag
  --saveEngine=<file>                Save the serialized engine
  --loadEngine=<file>                Load a serialized engine
  --getPlanVersionOnly               Print TensorRT version when loaded plan was created. Works without deserialization of the plan.
                                     Use together with --loadEngine. Supported only for engines created with 8.6 and forward.
  --tacticSources=tactics            Specify the tactics to be used by adding (+) or removing (-) tactics from the default 
                                     tactic sources (default = all available tactics).
                                     Note: Currently only cuDNN, cuBLAS, cuBLAS-LT, and edge mask convolutions are listed as optional
                                           tactics.
                                   Tactic Sources: tactics ::= [","tactic]
                                                     tactic  ::= (+|-)lib
                                                   lib     ::= "CUBLAS"|"CUBLAS_LT"|"CUDNN"|"EDGE_MASK_CONVOLUTIONS"
                                                               |"JIT_CONVOLUTIONS"
                                     For example, to disable cudnn and enable cublas: --tacticSources=-CUDNN,+CUBLAS
  --noBuilderCache                   Disable timing cache in builder (default is to enable timing cache)
  --noCompilationCache               Disable Compilation cache in builder, and the cache is part of timing cache (default is to enable compilation cache)
  --errorOnTimingCacheMiss           Emit error when a tactic being timed is not present in the timing cache (default = false)
  --timingCacheFile=<file>           Save/load the serialized global timing cache
  --preview=features                 Specify preview feature to be used by adding (+) or removing (-) preview features from the default
                                   Preview Features: features ::= [","feature]
                                                       feature  ::= (+|-)flag
                                                     flag     ::= "aliasedPluginIO1003"
                                                                  |"profileSharing0806"
  --builderOptimizationLevel         Set the builder optimization level. (default is 3)
                                     Higher level allows TensorRT to spend more building time for more optimization options.
                                     Valid values include integers from 0 to the maximum optimization level, which is currently 5.
  --maxTactics                       Set the maximum number of tactics to time when there is a choice of tactics. (default is -1)
                                     Larger number of tactics allow TensorRT to spend more building time on evaluating tactics.
                                     Default value -1 means TensorRT can decide the number of tactics based on its own heuristic.
  --hardwareCompatibilityLevel=mode  Make the engine file compatible with other GPU architectures. (default = none)
                                   Hardware Compatibility Level: mode ::= "none" | "ampere+"
                                         none = no compatibility
                                         ampere+ = compatible with Ampere and newer GPUs
  --runtimePlatform=platform         Set the target platform for runtime execution. (default = SameAsBuild)
                                     When this option is enabled, --skipInference is enabled by default.
                                   RuntimePlatfrom: platform ::= "SameAsBuild" | "WindowsAMD64"
                                         SameAsBuild = no requirement for cross-platform compatibility.
                                         WindowsAMD64 = set the target platform for engine execution as Windows AMD64 system
  --tempdir=<dir>                    Overrides the default temporary directory TensorRT will use when creating temporary files.
                                     See IRuntime::setTemporaryDirectory API documentation for more information.
  --tempfileControls=controls        Controls what TensorRT is allowed to use when creating temporary executable files.
                                     Should be a comma-separated list with entries in the format (in_memory|temporary):(allow|deny).
                                     in_memory: Controls whether TensorRT is allowed to create temporary in-memory executable files.
                                     temporary: Controls whether TensorRT is allowed to create temporary executable files in the
                                                filesystem (in the directory given by --tempdir).
                                     For example, to allow in-memory files and disallow temporary files:
                                         --tempfileControls=in_memory:allow,temporary:deny
                                     If a flag is unspecified, the default behavior is "allow".
  --maxAuxStreams=N                  Set maximum number of auxiliary streams per inference stream that TRT is allowed to use to run 
                                     kernels in parallel if the network contains ops that can run in parallel, with the cost of more 
                                     memory usage. Set this to 0 for optimal memory usage. (default = using heuristics)
  --profile                          Build with dynamic shapes using a profile with the min/max/opt shapes provided. Can be specified
                                         multiple times to create multiple profiles with contiguous index.
                                     (ex: --profile=0 --minShapes=<spec> --optShapes=<spec> --maxShapes=<spec> --profile=1 ...)
  --calibProfile                     Select the optimization profile to calibrate by index. (default = 0)
  --allowWeightStreaming             Enable a weight streaming engine. Must be specified with --stronglyTyped. TensorRT will disable
                                     weight streaming at runtime unless --weightStreamingBudget is specified.
  --markDebug                        Specify list of names of tensors to be marked as debug tensors. Separate names with a comma

=== Inference Options ===
  --shapes=spec               Set input shapes for dynamic shapes inference inputs.
                              Note: Input names can be wrapped with escaped single quotes (ex: 'Input:0').
                              Example input shapes spec: input0:1x3x256x256, input1:1x3x128x128
                              For scalars (0-D shapes), use input0:scalar or simply input0: with nothing after the colon.
                              Each input shape is supplied as a key-value pair where key is the input name and
                              value is the dimensions (including the batch dimension) to be used for that input.
                              Each key-value pair has the key and value separated using a colon (:).
                              Multiple input shapes can be provided via comma-separated key-value pairs, and each input 
                              name can contain at most one wildcard ('*') character.
  --loadInputs=spec           Load input values from files (default = generate random inputs). Input names can be wrapped with single quotes (ex: 'Input:0')
                            Input values spec ::= Ival[","spec]
                                         Ival ::= name":"file
                              Consult the README for more information on generating files for custom inputs.
  --iterations=N              Run at least N inference iterations (default = 10)
  --warmUp=N                  Run for N milliseconds to warmup before measuring performance (default = 200)
  --duration=N                Run performance measurements for at least N seconds wallclock time (default = 3)
                              If -1 is specified, inference will keep running unless stopped manually
  --sleepTime=N               Delay inference start with a gap of N milliseconds between launch and compute (default = 0)
  --idleTime=N                Sleep N milliseconds between two continuous iterations(default = 0)
  --infStreams=N              Instantiate N execution contexts to run inference concurrently (default = 1)
  --exposeDMA                 Serialize DMA transfers to and from device (default = disabled).
  --noDataTransfers           Disable DMA transfers to and from device (default = enabled).
  --useManagedMemory          Use managed memory instead of separate host and device allocations (default = disabled).
  --useSpinWait               Actively synchronize on GPU events. This option may decrease synchronization time but increase CPU usage and power (default = disabled)
  --threads                   Enable multithreading to drive engines with independent threads or speed up refitting (default = disabled) 
  --useCudaGraph              Use CUDA graph to capture engine execution and then launch inference (default = disabled).
                              This flag may be ignored if the graph capture fails.
  --timeDeserialize           Time the amount of time it takes to deserialize the network and exit.
  --timeRefit                 Time the amount of time it takes to refit the engine before inference.
  --separateProfileRun        Do not attach the profiler in the benchmark run; if profiling is enabled, a second profile run will be executed (default = disabled)
  --skipInference             Exit after the engine has been built and skip inference perf measurement (default = disabled)
  --persistentCacheRatio      Set the persistentCacheLimit in ratio, 0.5 represent half of max persistent L2 size (default = 0)
  --useProfile                Set the optimization profile for the inference context (default = 0 ).
  --allocationStrategy=spec   Specify how the internal device memory for inference is allocated.
                            Strategy: spec ::= "static", "profile", "runtime"
                                  static = Allocate device memory based on max size across all profiles.
                                  profile = Allocate device memory based on max size of the current profile.
                                  runtime = Allocate device memory based on the actual input shapes.
  --saveDebugTensors          Specify list of names of tensors to turn on the debug state
                              and filename to save raw outputs to.
                              These tensors must be specified as debug tensors during build time.
                            Input values spec ::= Ival[","spec]
                                         Ival ::= name":"file
  --weightStreamingBudget     Set the maximum amount of GPU memory TensorRT is allowed to use for weights.
                              It can take on the following values:
                                -2: (default) Disable weight streaming at runtime.
                                -1: TensorRT will automatically decide the budget.
                                 0-100%: Percentage of streamable weights that reside on the GPU.
                                         0% saves the most memory but will have the worst performance.
                                         Requires the % character.
                                >=0B: The exact amount of streamable weights that reside on the GPU. Supports the 
                                     following base-2 suffixes: B (Bytes), G (Gibibytes), K (Kibibytes), M (Mebibytes).

=== Reporting Options ===
  --verbose                   Use verbose logging (default = false)
  --avgRuns=N                 Report performance measurements averaged over N consecutive iterations (default = 10)
  --percentile=P1,P2,P3,...   Report performance for the P1,P2,P3,... percentages (0<=P_i<=100, 0 representing max perf, and 100 representing min perf; (default = 90,95,99%)
  --dumpRefit                 Print the refittable layers and weights from a refittable engine
  --dumpOutput                Print the output tensor(s) of the last inference iteration (default = disabled)
  --dumpRawBindingsToFile     Print the input/output tensor(s) of the last inference iteration to file(default = disabled)
  --dumpProfile               Print profile information per layer (default = disabled)
  --dumpLayerInfo             Print layer information of the engine to console (default = disabled)
  --dumpOptimizationProfile   Print the optimization profile(s) information (default = disabled)
  --exportTimes=<file>        Write the timing results in a json file (default = disabled)
  --exportOutput=<file>       Write the output tensors to a json file (default = disabled)
  --exportProfile=<file>      Write the profile information per layer in a json file (default = disabled)
  --exportLayerInfo=<file>    Write the layer information of the engine in a json file (default = disabled)

=== System Options ===
  --device=N                  Select cuda device N (default = 0)
  --useDLACore=N              Select DLA core N for layers that support DLA (default = none)
  --staticPlugins             Plugin library (.so) to load statically (can be specified multiple times)
  --dynamicPlugins            Plugin library (.so) to load dynamically and may be serialized with the engine if they are included in --setPluginsToSerialize (can be specified multiple times)
  --setPluginsToSerialize     Plugin library (.so) to be serialized with the engine (can be specified multiple times)
  --ignoreParsedPluginLibs    By default, when building a version-compatible engine, plugin libraries specified by the ONNX parser 
                              are implicitly serialized with the engine (unless --excludeLeanRuntime is specified) and loaded dynamically. 
                              Enable this flag to ignore these plugin libraries instead.

=== Help ===
  --help, -h                  Print this message
```