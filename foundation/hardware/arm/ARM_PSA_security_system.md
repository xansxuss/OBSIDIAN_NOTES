# 🛡️ ARM PSA Security System 完整解析

_(工程友善，實務導向，略帶 Z 世代科學家 vibes)_

ARM PSA 是 ARM 推出的 **全棧安全架構標準**，目的是讓 IoT / Edge SoC 不會「各家各做自己的安全 Frankenstein」。  
它不是某個硬體，而是一整套：

- 威脅模型（Threat Model）
    
- 安全需求（Security Requirements）
    
- 系統架構（System Architecture）
    
- API 標準（主要是 PSA Certified APIs）
    
- 評測與認證流程（PSA Certification Level 1~3）
    

換句話說：  
**你想設計安全 SoC? 跟著 PSA 做，不會歪。**  
**你想做 TEE / Secure Boot / Key Storage? PSA 有菜單。**

---

# 🧩 PSA 架構三大塊：

## 1. **PSA Threat Model & Security Requirements (TMSR)**

ARM 先把 IoT/Edge 可能遇到的攻擊寫成 SRS：

- 側信道（SCA）
    
- Fault Injection
    
- Secure Boot 破壞
    
- Cloning / Key Extraction
    
- Debug Interface 攻擊
    
- Firmware Tampering
    
- OTA 攻擊
    

這份 TMSR 是你做 secure SoC 時的**安全規格書**。

---

## 2. **PSA Firmware Architecture（重點）**

對你的系統工程設計最有幫助的就是這個。

### PSA 架構由兩層世界組成：

### 🔐 **Secure Processing Environment (SPE)**

通常跑在 **ARM TrustZone Secure World** 或 Secure Element (SE)。

負責：

- Secure Boot 執行鏈（Root of Trust）
    
- Key Management
    
- Crypto Services (PSA Crypto API)
    
- Secure Storage
    
- Attestation Service (PSA Attestation API)
    
- Firmware Update Service (PSA Firmware Update API)
    

### 🌐 **Non-secure Processing Environment (NSPE)**

使用者 app、IoT 設備邏輯、OS（Linux / RTOS）。

兩者透過 **PSA Firmware Framework API（FF-M）** 溝通。  
如果你做 Jetson、Qualcomm、NXP SoC 之類，這就是在 TrustZone Secure OS 上跑的 PSA Services。

---

## 3. **PSA Certification（1~3 級）**

像安全版 TOEIC。

### **Level 1：文件 + 威脅模型**

你只要寫出 Threat Model + 設計符合 PSA。

### **Level 2：抵禦 Software 攻擊**

需要 Secure Boot、Crypto、TEE 服務都有正確防護。

### **Level 3：抵禦 Hardware 攻擊**

要能防住：

- SCA
    
- Glitching/Fault
    
- Physical probing
    

大部分 MCU 達不到 Level 3，要用 SE / Crypto Island / HSM。

---

# 🧱 PSA 技術模組（與你會遇到的硬體對應）

|PSA Component|實際硬體對應|功能|
|---|---|---|
|**PSA Root of Trust (PRoT)**|TrustZone Secure World + immutable ROM + OTP|可信起點|
|**PSA Crypto API**|CryptoCell / CC312 / CC710 / Cryptoisland / HSM|AES、ECC、HKDF、TRNG|
|**PSA Attestation API**|SoC Unique Key + RoT|Device identity & claim|
|**PSA Secure Storage**|TrustZone FS + RPMB + PUF key storage|安全 Key/Blob storage|
|**PSA FF-M**|SCMI + SMC/IPC 通訊|NS ↔ Secure OS 安全呼叫|
|**PSA Update**|Secure Bootloader + anti-rollback|安全 OTA|

---

# 📦 PSA 與 TrustZone / CryptoCell / Crypto Island 的關係

### PSA 是「指南」

TrustZone / CryptoCell / Crypto Island 是「設備」。

- **TrustZone**：跑 Secure World（TEE）
    
- **CryptoCell**：硬體加速器 + Key Vault
    
- **Crypto Island**：大型 SoC 的獨立 HSM（更硬核）
    

PSA 說：

> 要做安全系統，就用 TrustZone 跑 TEE + CryptoCell 做 Crypto + Secure Boot + Secure Storage。

也就是 **把分散的安全硬體統一成一個完整模型**。

---

# 🏗️ PSA 典型系統框架（超實用）

``` bash
+--------------------------------------------+
| Non-secure World (Linux / RTOS)            |
|   - Apps                                   |
|   - Drivers                                |
+-----------------------+--------------------+
                        |
                        | FF-M API (IPC/SMC)
                        v
+--------------------------------------------+
| Secure World (PSA RoT)                     |
|  - Secure Boot                              |
|  - Crypto Service (PSA Crypto API)          |
|  - Secure Storage                           |
|  - Attestation                              |
|  - Firmware Update                           |
|  - Secure IPC                                |
+-----------------------+--------------------+
                        |
                        v
+--------------------------------------------+
| Hardware Root of Trust                      |
|   - Immutable ROM                           |
|   - Unique Device Key                       |
|   - CryptoCell / SE / CryptoIsland          |
|   - TRNG                                    |
+--------------------------------------------+
```

---

🚀 PSA 對工程師最有用的東西：
1. PSA Crypto API（你能在 MbedTLS/TF-M 上直接用）
例如：
`psa_generate_key()`, `psa_asymmetric_sign()`, `psa_cipher_encrypt()`

這也是 IoT 裝置的“跨廠家標準 Crypto API”。

2. TF-M (Trusted Firmware-M)
ARM 官方開源實作的 PSA Secure World。

如果你設計自己的 MCU / IoT SoC：

TF-M 你只要 port 一下平台層

Secure Boot / Secure Storage / Attestation 全部有現成模組

跟你的 Secure Boot chain 打通即可

