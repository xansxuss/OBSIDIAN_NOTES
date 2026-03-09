## 🛠️ onnx2tf 工具參數對照表

### 1. 基本輸入與輸出 (Basic I/O)

|**參數 (長/短)**|**說明**|**預設值 / 備註**|
|---|---|---|
|`-i`, `--input_onnx_file_path`|**必填**。輸入的 ONNX 模型路徑。|-|
|`-o`, `--output_folder_path`|輸出的資料夾路徑。|`saved_model`|
|`-V`, `--version`|顯示版本資訊並退出。|-|
|`-oh5`, `--output_h5`|輸出 Keras (HDF5) 格式模型。|-|
|`-okv3`, `--output_keras_v3`|輸出 Keras v3 格式模型。|-|
|`-otfv1pb`, `--output_tfv1_pb`|輸出 TensorFlow v1 (.pb) 格式模型。|-|
|`-ow`, `--output_weights`|僅輸出權重為 HDF5 格式。|-|

---
### 2. 量化相關 (Quantization)

|**參數 (長/短)**|**說明**|**預設值 / 備註**|
|---|---|---|
|`-odrqt`, `--output_dynamic_range_quantized_tflite`|輸出**動態範圍量化** (Dynamic Range Quantization) 的 TFLite。|-|
|`-oiqt`, `--output_integer_quantized_tflite`|輸出**全整數向量化** (Integer Quantized) 的 TFLite。|-|
|`-qt`, `--quant_type`|選擇量化粒度：`per-channel` 或 `per-tensor`。|`per-channel`|
|`-iqd`, `--input_quant_dtype`|全整數量化時的輸入型別：`int8`, `uint8`, `float32`。|`int8`|
|`-oqd`, `--output_quant_dtype`|全整數量化時的輸出型別：`int8`, `uint8`, `float32`。|`int8`|
|`-cind`, `--custom_input_op_name_np_data_path`|指定量化或驗證用的自定義 Numpy 資料、Mean 與 Std。|用於校準資料輸入|

---
### 3. 模型形狀與轉換校正 (Shape & Transformation)

|**參數 (長/短)**|**說明**|**預設值 / 備註**|
|---|---|---|
|`-b`, `--batch_size`|固定動態 Batch Size 為指定數值。|須 $\ge 1$|
|`-ois`, `--overwrite_input_shape`|強制覆寫輸入 Shape (格式：`name:d1,d2...`)。|會覆蓋 `-b`|
|`-sh`, `--shape_hints`|為動態維度提供推理暗示，不改變模型結構。|用於 `-coto`|
|`-k`, `--keep_ncw_or_nchw_or_ncdhw_input_names`|指定輸入名稱維持 **Channel-First** 格式 (如 NCHW)。|適用 3D/4D/5D|
|`-kt`, `--keep_nwc_or_nhwc_or_ndhwc_input_names`|指定輸入名稱維持 **Channel-Last** 格式 (如 NHWC)。|適用 3D/4D/5D|
|`-kat`, `--keep_shape_absolutely_input_names`|無條件維持原始輸入 Shape 的 OP 名稱。|-|
|`-dsm`, `--disable_strict_mode`|**關閉嚴格模式**。可大幅加速，但易產生維度置換錯誤或精確度問題。|預設為開啟|
|`-nuo`, `--not_use_onnxsim`|禁用 `onnx-simplifier`。關閉此項極易導致轉換失敗。|-|

---
### 4. 運算元優化與取代 (Operators Optimization)

|**參數 (長/短)**|**說明**|**備註**|
|---|---|---|
|`-ofgd`, `--optimization_for_gpu_delegate`|盡量將運算元替換為支援 TFLite GPU Delegate 的版本。|強烈建議 GPU 部署使用|
|`-dgc`, `--disable_group_convolution`|禁用 GroupConv，改用 SeparableConv 代替。|-|
|`-eru`, `--enable_rnn_unroll`|展開 RNN 循環（提升速度但增加記憶體消耗）。|-|
|`-ebu`, `--enable_batchmatmul_unfold`|將 BatchMatMul 拆解為原始的 MatMul。|-|
|`-rtpo`, `--replace_to_pseudo_operators`|將特定 OP (如 GeLU, PReLU) 替換為虛擬算子。|-|
|`-rari64` / `-rarf32`|將 ArgMax 替換為 ReduceMax (輸出 int64 或 float32)。|-|

---
### 5. 驗證與調錯 (Verification & Debugging)

| **參數 (長/短)**                                               | **說明**                                     | **預設值 / 備註**    |
| ---------------------------------------------------------- | ------------------------------------------ | --------------- |
| `-coto`, `--check_onnx_tf_outputs_elementwise_close`       | 檢查最終輸出的 ONNX 與 TF 結果是否一致。                  | -               |
| `-cotof`, `--check_onnx_tf_outputs_elementwise_close_full` | 逐一檢查**所有**層的輸出是否一致。                        | 極度耗時            |
| `-agj`, `--auto_generate_json`                             | **自動生成調優 JSON**。自動尋找最佳轉換參數以消除誤差。           | 不能與 `-cotof` 共用 |
| `-cgdc`, `--check_gpu_delegate_compatibility`              | 使用 TFLite Analyzer 檢查 Float16 模型是否支援 GPU。  | -               |
| `-v`, `--verbosity`                                        | 設定記錄等級 (`debug`, `info`, `warn`, `error`)。 | `debug`         |
| `-n`, `--non_verbose`                                      | 簡化輸出，等同於 `--verbosity error`。              | -               |