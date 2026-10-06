---
title: LLM LoRA 微調：打造 Python 與 C/C++ 專用模型（含非 PRC 模型選擇與 NVIDIA NIM 部署）
source: 
author: 
published: 
created: 2026-10-05
description: 整理 token 概念、用 LoRA 微調程式專用 LLM 的完整流程、非 PRC 來源的開放權重模型清單，以及用 NVIDIA NIM 部署 adapter 的方式。
categorization: AI_system/multimodal/generate_language_model
tags:
  - LLM
  - LoRA
  - QLoRA
  - fine-tuning
  - tokenizer
  - code-model
  - NVIDIA-NIM
  - open-weight-models
---

# LLM LoRA 微調：打造 Python 與 C/C++ 專用模型

## 1. 基礎概念：Token

- Token 是 LLM 處理文字的最小單位，文字先被切開，再對應成整數 ID，模型只處理數字。
- 一個 token 不一定是一個字，可能是單字、單字片段、標點或空白。
- 粗估：英文約 1 token ≈ 4 字元 ≈ 0.75 單字；中文約 1 字 ≈ 1～2 tokens（依模型而異）。
- 常見切分演算法是 BPE（Byte Pair Encoding），詳見 [[BPE]] 與 [[Tokenizer]]。

```
文字 → Tokenizer → Token ID → Embedding → 模型運算 → 預測下一個 token → 還原成文字
```

### Token 的實際用途

| 用途 | 說明 |
|---|---|
| 計費 | API 依輸入與輸出 token 數量收費 |
| Context window | 單次可處理的 token 上限，含輸入與輸出 |
| 控制輸出 | `max_tokens` 限制最多生成幾個 token |
| 效能指標 | 生成速度用 tokens/s 衡量 |

> [!warning] Tokenizer 必須與基底模型一致
> 訓練與推論都要使用該模型自己的 tokenizer 與 chat template，不可混用。

## 2. 先選對做法

| 做法 | 說明 | 門檻 |
|---|---|---|
| RAG | 不改模型，把文件切塊存入向量資料庫，提問時檢索 | 低 |
| Fine-tuning（LoRA / QLoRA） | 在現成模型上繼續訓練，調整風格、格式與領域能力 | 中 |
| Pre-training | 從隨機權重開始訓練 | 極高，個人幾乎不可行 |

**核心觀念**：LoRA 是在現成模型上做小幅調整，不是從零教會寫程式。要得到強的程式模型，第一步是選一個**本來就會寫程式**的基底模型。相關：[[LoRA]]、[[Quantization]]。

## 3. 微調完整流程

```
收集程式碼 → 過濾清理 → 轉成訓練格式 → LoRA 訓練 → 自動化評估（編譯＋測試）→ 迭代
```

### 3.1 資料準備

三類資料混合使用：

1. **指令型**（instruction → code）：最有效。
2. **自己的專案程式碼**：學到命名、風格與架構。
3. **FIM（Fill-In-the-Middle）**：給前後文補中間，適合程式補全。

要點：

- 只使用有權使用的程式碼（自己的、授權寬鬆的開源專案）。
- 去重、移除自動生成檔案、移除金鑰與密碼。
- 想讓模型習慣特定風格（例如 C++ 不用標準函式庫），資料中要大量出現，並在 instruction 明確寫限制。
- 品質優先於數量，數百到數千筆高品質資料就能看到效果。

### 3.2 資料格式

對話型（instruct 模型，使用 `messages`）：

```json
{"messages": [
  {"role": "user", "content": "用 C 寫一個 ring buffer，不使用標準函式庫的 malloc。"},
  {"role": "assistant", "content": "```c\n...程式碼...\n```"}
]}
```

base 模型沒有 chat template，改用 prompt／completion：

```json
{"prompt": "# 用 Python 寫一個二分搜尋函式\n", "completion": "def binary_search(arr, target):\n    ..."}
```

### 3.3 LoRA 訓練設定（程式任務調整版）

```python
from peft import LoraConfig
from trl import SFTConfig

lora = LoraConfig(
    r=32,                        # 附加矩陣的秩，程式任務可試 16~64
    lora_alpha=64,               # 通常設為 r 的 2 倍
    lora_dropout=0.05,
    target_modules="all-linear", # 不同模型層名稱不同，all-linear 最保險
    task_type="CAUSAL_LM",
)

args = SFTConfig(
    output_dir="out",
    num_train_epochs=2,               # 資料少時 2~3 輪，過多會過擬合
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,   # 用時間換記憶體，等效 batch size 16
    learning_rate=1e-4,               # 偏保守，避免洗掉原有能力
    max_length=2048,                  # 程式碼較長
    eval_strategy="steps",
    eval_steps=50,
    bf16=True,
)
```

- 學習率調低是為了避免 catastrophic forgetting（災難性遺忘）。
- 觀察 `eval_loss`：訓練 loss 下降但驗證 loss 上升，就是過擬合。
- 使用 QLoRA 時，基底模型以 4-bit 載入，可大幅降低 VRAM 需求。

### 3.4 驗證模型是否真的變強

程式碼可以自動驗證對錯：

1. 留 30～50 題不參與訓練的測試題，附單元測試或預期輸出。
2. 訓練前先測基底模型，記錄基準分數。
3. 訓練後用同一份題目重測，沒進步就代表微調無效。
4. 可參考 HumanEval、MBPP（偏 Python），確認通用能力沒有退步。

C/C++ 編譯＋執行檢查工具：

```python
import subprocess, tempfile, os

def check_c_code(code: str, compiler: str = "gcc", timeout: int = 5):
    """編譯並執行 C 程式碼，回傳 (編譯成功, 正常執行, 輸出)"""
    with tempfile.TemporaryDirectory() as d:
        src, exe = os.path.join(d, "t.c"), os.path.join(d, "t")
        with open(src, "w") as f:
            f.write(code)
        build = subprocess.run([compiler, "-Wall", "-Wextra", "-O0", src, "-o", exe],
                               capture_output=True, text=True)
        if build.returncode != 0:
            return False, False, build.stderr
        try:
            run = subprocess.run([exe], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:   # 防止無窮迴圈
            return True, False, "timeout"
        return True, run.returncode == 0, run.stdout
```

> [!danger] 安全
> 執行模型生成的程式碼有風險，請在虛擬機、Docker 或沙盒中進行。

### 3.5 常見陷阱

- 資料少又訓練太多輪 → 只會背資料。
- 混入有 bug 的程式碼 → 模型學到錯誤寫法。
- tokenizer 或 chat template 不一致 → 輸出異常。
- 期望過高：1.5B～7B 的模型經 LoRA 能明顯提升特定風格與領域，但整體推理仍受基底模型上限限制。

## 4. 非 PRC 來源的開放權重模型

> [!note] 選擇原則
> 不使用中國來源模型（Qwen、DeepSeek、GLM、Kimi、MiniMax 等）。distillation（蒸餾）相關指控多半未有定論，務實做法是選**訓練資料來源透明、授權清楚**的模型。

| 開發者（國家） | 代表模型與大小 | 授權 | 程式能力 | LoRA 微調門檻 |
|---|---|---|---|---|
| Mistral AI（法國） | Devstral Small 2（24B 稠密）、Mistral Small 4（119B MoE） | Apache 2.0 | 強，Devstral 專為程式與軟體工程設計 | 24B 需 24GB 以上；119B 需多卡 |
| Google（美國） | Gemma 4 31B、Gemma 4 26B A4B（MoE）、Gemma 3 4B / 27B | Gemma 條款或 Apache 2.0（各來源說法不一，待確認） | 中上 | 26～31B 需 24GB 以上；4B 門檻低 |
| Meta（美國） | Llama 3.2（1B、3B）、Llama 3.3（70B）、Llama 4 Scout | Llama 社群授權（月活躍使用者低於 7 億可商用） | 一般到中等 | 1～3B 最低；70B 需多卡 |
| NVIDIA（美國） | Nemotron 3 Nano（約 31.6B，MoE）、Nano 4B、Super（120B） | NVIDIA 自家開放模型授權 | 中上，與 [[NVIDIA NIM]] 整合順 | 4B 低；30B MoE 的 LoRA 工具支援待確認 |
| IBM（美國） | Granite 3.3 8B、Granite Code 34B | Apache 2.0 | Granite Code 為程式專用 | 8B 中等；34B 偏高 |
| OpenAI（美國） | gpt-oss-20b、gpt-oss-120b | Apache 2.0 | 中上 | 20B 約需 24GB 級顯卡 |
| Microsoft（美國） | Phi-4-mini | 以官方頁面為準 | 小模型，適合學習 | 低 |
| AI2（美國） | OLMo（資料、程式碼、tokenizer 全公開） | 開放 | 程式能力非強項 | 低到中 |
| BigCode（國際合作） | StarCoder2（3B、7B、15B） | BigCode OpenRAIL-M | 專為程式，資料經授權篩選 | 3B～7B 低 |

> [!warning] 閱讀注意
> 1. 模型資訊更新很快，授權欄位各來源說法不一，使用前以 Hugging Face 官方頁面為準。
> 2. 「推論需要的 VRAM」不等於「LoRA 訓練需要的 VRAM」，訓練還需放梯度、優化器狀態與較長上下文。
> 3. StarCoder2 一列為既有知識，非本次搜尋結果，需另行確認。

### 依 VRAM 的起點建議

| VRAM | 建議起點 |
|---|---|
| 約 8～12GB | Granite 3.3 8B、Gemma 3 4B、StarCoder2 3B～7B、Nemotron 3 Nano 4B |
| 24GB 以上 | Devstral Small 2 或 Gemma 4 中型版本 |
| 需搭配 NIM | 優先 Llama 或 Nemotron，先在 NVIDIA 目錄確認支援 |

## 5. NVIDIA NIM：部署 LoRA adapter

- NIM（NVIDIA Inference Microservices）是**部署與推論平台**，不負責訓練。
- 流程：訓練（Hugging Face PEFT 或 NeMo）→ 產生 LoRA adapter → 放進 NIM → 用 API 呼叫。相關：[[TensorRT]]、[[Docker]]。
- LoRA 權重綁定特定基底模型，**NIM 服務的模型必須與訓練時的基底模型相同**。
- 基底模型必須是 NIM 有支援的模型。

### 運作方式

- 設定環境變數 `NIM_PEFT_SOURCE` 指向 adapter 資料夾，`NIM_PEFT_REFRESH_INTERVAL` 設定偵測間隔。
- 複製 adapter 資料夾進去即載入，移除即卸載。
- 用 `/v1/models` 查詢可用的 adapter ID，請求的 `model` 欄位填 adapter ID。
- 文件提到可選擇 `-feat_lora` 設定檔（視模型而定）。

```
my_adapters/
└── my_code_lora/                # 資料夾名稱即 adapter ID
    ├── adapter_config.json
    └── adapter_model.safetensors
```

```python
import requests

URL = "http://localhost:8000/v1"   # NIM 容器位址，依部署調整

# 1. 確認 adapter 已載入
for m in requests.get(f"{URL}/models").json()["data"]:
    print(m["id"])

# 2. 把 adapter ID 填進 model 欄位
payload = {
    "model": "my_code_lora",
    "messages": [{"role": "user", "content": "用 C 寫一個固定容量的 ring buffer。"}],
    "max_tokens": 512,
    "temperature": 0.2,            # 寫程式用低溫度，輸出較穩定
}
resp = requests.post(f"{URL}/chat/completions", json=payload).json()
print(resp["choices"][0]["message"]["content"])
```

### 注意事項

- **訓練格式與推論端點要一致**：用 `messages` 格式訓練，就用 `/v1/chat/completions` 呼叫，否則訓練與推論不一致會導致非預期輸出。
- 部署前要充分評估（沿用 3.4 的編譯＋測試）。
- adapter rank 不必過大，rank 越大佔記憶體越多，品質提升不一定成比例。
- QLoRA 訓練（4-bit）與 NIM 部署（通常較高精度）精度不同，可能有些落差，部署後需重新評估（個人經驗判斷，非官方說法）。
- 需要 NVIDIA GPU、Docker 與 NVIDIA Container Toolkit；商用授權條款請自行至官網確認。

## 6. 待辦與待確認

- [ ] 確認訓練用 GPU 型號與 VRAM，決定基底模型大小
- [ ] 到 NVIDIA 模型目錄確認哪些非 PRC 模型支援 LoRA
- [ ] 整理自己的專案程式碼，產生 instruction 資料並抽查
- [ ] 建立 30～50 題測試集，先測基底模型取得基準分數
- [ ] 相關筆記：[[Unsloth]]、[[Axolotl]]、[[llama.cpp]]（微調後轉 GGUF 推論）

## 參考資料

- NVIDIA NIM for LLMs：Fine-Tuning with LoRA（docs.nvidia.com/nim/large-language-models/latest/advanced-use-cases/finetune-lora.html）
- NVIDIA NIM for LLMs：Parameter-Efficient Fine-Tuning（docs.nvidia.com/nim/large-language-models/latest/peft.html）
- Best Local LLM for Coding in 2026（vdf.ai/blog/best-local-llm-for-coding/）
- Open-Source LLMs for Developers（codetocloud.io/blog/open-source-llms-developers/）
