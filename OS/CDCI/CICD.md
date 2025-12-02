# 🧨 什麼是 CI/CD？（一句話版）

- **CI（Continuous Integration）**：  
    你每 push 一次，系統就幫你**自動測、Lint、Build、跑工具鏈**，確定不會把專案搞掛。
- **CD（Continuous Delivery/Deployment）**：  
    你每 build 出產物，系統就幫你**自動部署**（或準備好一鍵部署）。

簡單講：  
**CI = 保證你沒炸彈**  
**CD = 把結果丟到戰場**

---

# 🧩 CI/CD 會做哪些事情？

以下是你日常工程會遇到的：

## 🔧 **CI 工作**

- 編譯（C++/CUDA/Python/Android…）
- 跑單元測試
- Lint / 格式化（clang-format, black）
- 靜態分析（clang-tidy, cppcheck）
- Docker build
- 多平台 build（x86/ARM/GPU）
- 檔案簽章（release artifact 簽名）

## 🚀 **CD 工作**

- 部署 Docker image → k8s
- 上線新版本到 VM / Bare metal
- OTA edge device（Jetson / IPC）
- CI build 的 artifacts 丟到 release / artifact store

---

# 🏛️ CI/CD 架構比較（重點講你工程師角度 care 的）

## 1️⃣ GitHub Actions / GitLab CI

**雲端托管，不用自己養一堆 server**

|項目|GitHub Actions / GitLab CI|
|---|---|
|建置成本|超低|
|維護成本|超低|
|介面|YAML|
|可擴展性|靠 self-hosted runner|
|可否跑 GPU|要自己架 runner|
|適合誰|大多數團隊、AI 工程、C++ 專案|

**優點**

- 零維護
- 與 repo 深度整合
- YAML pipeline 很好讀
- 自架 runner → ARM、Jetson、GPU 都能跑
- Artifact / cache 多功能

**缺點**

- GitHub 雲端 runner 無 GPU
- 重度定制就有限

---

## 2️⃣ Jenkins

老牌，但依然是地表上最萬能的 CI/CD。

|項目|Jenkins|
|---|---|
|建置成本|高|
|維護成本|高（插件地獄）|
|可擴展性|無上限|
|可否跑 GPU|超行|
|適合誰|需要超客製化的企業|

**優點**

- 真的無所不能
- GPU build、distributed build、cross-compile 都能玩
- pipeline 可 code 化

**缺點**

- 你得自己養
- 升級、備份、plugin 相依性 → 噩夢
- 運維成本高

> 必須說：如果你團隊沒有專職 DevOps，建議不要選 Jenkins。

---

## 3️⃣ Buildkite

很香的大型企業愛用 CI/CD（Shopify、Uber、Pinterest 都用）

|項目|Buildkite|
|---|---|
|建置成本|中到高（需要 agents）|
|維護成本|低（比 Jenkins 爽太多）|
|可擴展性|極強|
|適合誰|需要高效能 build farm 的工程團隊|

**優點**

- Control plane 給你
- Build 全在你的機器上（GPU farm / ARM farm）
- 超級好 scale
- 適合 C++、CUDA、TensorRT、大型專案

**缺點**

- 要錢
- YAML/Plugin 生態不及 GitHub CI

---

## 4️⃣ ArgoCD / FluxCD（只做 CD）

如果你有 **K8s**，CD 最佳解。

|項目|ArgoCD / FluxCD|
|---|---|
|用途|專門做 CD|
|位置|K8s 內|
|模式|GitOps|
|適合|雲端微服務|

---

# 📦 Build Server（編譯伺服器）怎麼選？

這是你的專業範圍了（AI + CUDA + C++）：

如果你會有：

- CUDA kernel build
- TensorRT engine build
- OpenCV CUDA build
- cross-compile to ARM（Jetson / RK3588 / embedded）

→ **你一定需要自架 build server**

常見做法：

### 🏭 Build farm 架構

``` bash
GitHub/GitLab CI    
	↓ 送 job 
Self-hosted runners    
	├── x86 高速 build    
	├── GPU CUDA/TensorRT build    
	├── ARM build（QEMU or 實機）    
	└── Docker build node    
	↓ 上傳 artifacts
```

這種方式彈性最大。

---

# 🧠 怎麼選？（簡單粗暴選型指南）

## 👉 你是小團隊 / 個人 / Startup

**選：GitHub Actions + 自架 runner**  
最省腦。功能也超夠用。

## 👉 你是 AI 團隊 / C++ 影像團隊

**選：GitHub/GitLab CI + GPU build server**  
你所有 CUDA / TRT build 全可自動化。  
方便做多平台 pipeline。

## 👉 你是企業級，大量專案，全自動化

**選：Buildkite**  
CI/CD 都能 scale，管理起來比 Jenkins 爽太多。

## 👉 有大量 legacy / 超客製需求

**選：Jenkins**  
但要注意你要有 DevOps 才撐得住。

## 👉 你在玩 k8s 微服務

**選：ArgoCD（CD） + 任意 CI**

---

# 🔥 快速總結（不廢話版）

|需求|建議|
|---|---|
|省事、快速上線|GitHub Actions|
|GPU build / CUDA pipeline|GitHub/GitLab self-hosted runners|
|AI/C++ 大型專案 build farm|Buildkite|
|公司老專案很多，需要客製化|Jenkins|
|K8s 服務部署|ArgoCD|