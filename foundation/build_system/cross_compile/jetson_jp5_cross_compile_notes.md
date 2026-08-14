# Jetson JetPack 5(R35.6 / Xavier NX)CUDA 11.4 交叉編譯環境建置筆記

> 目標：在 x86_64 host（Ubuntu，容器內）上，同時保留 CUDA 12.6 原生環境，
> 並建置一套獨立、可交叉編譯給 Jetson Xavier NX（JetPack 5.1.4 / L4T R35.6.0）
> 用的 CUDA 11.4 + cuDNN 8.6.0 + TensorRT 8.5.2 開發環境。

---

## 目錄

1. [Host 端 CUDA 11.4 / 12.6 雙版本並存](#1-host-端-cuda-114--126-雙版本並存)
2. [Jetson 交叉編譯環境建置概覽](#2-jetson-交叉編譯環境建置概覽)
3. [Sysroot 建置：展開 target 用 deb 套件](#3-sysroot-建置展開-target-用-deb-套件)
4. [四個核心踩坑紀錄](#4-四個核心踩坑紀錄)
5. [最終可用的 toolchain file](#5-最終可用的-toolchain-file)
6. [Orin / R36 對照：為什麼同樣做法在新版上更簡單](#6-orin--r36-對照為什麼同樣做法在新版上更簡單)

---

## 1. Host 端 CUDA 11.4 / 12.6 雙版本並存

**原則**：不要用 `apt` 裝 CUDA（會搶 `/usr/local/cuda` 全域連結），改用官方 **runfile**，
安裝到各自獨立的資料夾。

```bash
# CUDA 11.4
sudo sh cuda_11.4.4_470.82.01_linux.run --silent --toolkit \
    --toolkitpath=/usr/local/cuda-11.4

# CUDA 12.6
sudo sh cuda_12.6.0_560.28.03_linux.run --silent --toolkit \
    --toolkitpath=/usr/local/cuda-12.6
```

cuDNN 用 **tar 壓縮檔**手動複製進對應版本資料夾（不要用 deb，deb 會裝到系統共用路徑，
兩個版本會互相覆蓋）：

```bash
tar -xf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
sudo cp cudnn-*-archive/include/*.h /usr/local/cuda-11.4/include/
sudo cp cudnn-*-archive/lib/*.so*   /usr/local/cuda-11.4/lib64/

tar -xf cudnn-linux-x86_64-9.5.0.50_cuda12-archive.tar.xz
sudo cp cudnn-*-archive/include/*.h /usr/local/cuda-12.6/include/
sudo cp cudnn-*-archive/lib/*.so*   /usr/local/cuda-12.6/lib64/
```

**切換方式**：不要動全域 `/usr/local/cuda` 符號連結，改用 per-shell 的環境變數腳本：

```bash
# ~/env/cuda114.sh
export CUDA_HOME=/usr/local/cuda-11.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

`source ~/env/cuda114.sh` 或 `source ~/env/cuda126.sh`，兩個終端機視窗可以同時開、互不干擾。

---

## 2. Jetson 交叉編譯環境建置概覽

透過 SDK Manager 下載的檔案分三類：

| 分類 | 範例檔案 | 用途 |
|---|---|---|
| **Host（x86_64）端** | `cuda-repo-cross-aarch64-ubuntu2004-11-4-local_*.deb` | 裝進 host 自己的 `/usr/local/cuda-11.4/targets/aarch64-linux/`，給 nvcc 找 CUDA 交叉標頭檔用 |
| **Target（arm64）端** | `cudnn-local-tegra-repo-*_arm64.deb`、`nv-tensorrt-local-repo-*_arm64.deb` | **不能** `dpkg -i`（架構不符），要展開進 **sysroot** |
| **BSP／根檔案系統** | `Jetson_Linux_R35.6.0_aarch64.tbz2`、`Tegra_Linux_Sample-Root-Filesystem_R35.6.0_aarch64.tbz2` | 建置 sysroot 的骨架 |

### CUDA cross 套件（host 端）安裝

SDK Manager 下載的是 local repo deb：

```bash
sudo dpkg -i cuda-repo-cross-aarch64-ubuntu2004-11-4-local_11.4.19-1_all.deb
sudo apt-key add /var/cuda-repo-cross-aarch64-ubuntu2004-11-4-local/*.pub
sudo apt-get update
sudo apt-get install cuda-cross-aarch64-11-4
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu   # 交叉編譯器本體
```

裝完後會在**容器/host 自己**的 `/usr/local/cuda-11.4/targets/aarch64-linux/` 底下出現
CUDA 交叉標頭檔跟函式庫，跟 `/usr/local/cuda-11.4/targets/x86_64-linux/`（host 原生版本）並列。

> ⚠️ **注意**：上面裝的 `gcc-aarch64-linux-gnu` 這套交叉編譯器版本太新（GCC 11.4），
> 後面會在 [問題一](#問題一__malloc__-does-not-take-arguments) 炸開，最終改用 Bootlin GCC 9.3。

---

## 3. Sysroot 建置：展開 target 用 deb 套件

Target（arm64）用的套件本質上是「repo 定義包」，裡面還包著真正的內層 `.deb`，
必須先 `dpkg -x` 展開外層，再展開內層真正需要的套件進 sysroot。

```bash
SYSROOT=/sysroot/jetson-sysroot_r3560
DEBDIR=/mnt/storage_1/arm64/platfrom/jetson/Jetson_R35_6_0

# 1. 展開外層 repo deb，找出內層真正的 .deb 清單
mkdir -p /tmp/repo-extract/cudnn /tmp/repo-extract/trt
dpkg -x $DEBDIR/cudnn-local-tegra-repo-ubuntu2004-8.6.0.166_1.0-1_arm64.deb /tmp/repo-extract/cudnn
dpkg -x $DEBDIR/nv-tensorrt-local-repo-l4t-8.5.2-cuda-11.4_1.0-1_arm64.deb /tmp/repo-extract/trt

# 2. 只展開交叉編譯連結真正需要的內層套件（跳過 Python 綁定、範例、轉換工具、meta 套件）
CUDNN_DIR=/tmp/repo-extract/cudnn/var/cudnn-local-tegra-repo-ubuntu2004-8.6.0.166
TRT_DIR=/tmp/repo-extract/trt/var/nv-tensorrt-local-repo-l4t-8.5.2-cuda-11.4

dpkg -x $CUDNN_DIR/libcudnn8_8.6.0.166-1+cuda11.4_arm64.deb $SYSROOT
dpkg -x $CUDNN_DIR/libcudnn8-dev_8.6.0.166-1+cuda11.4_arm64.deb $SYSROOT

dpkg -x $TRT_DIR/libnvinfer8_8.5.2-1+cuda11.4_arm64.deb $SYSROOT
dpkg -x $TRT_DIR/libnvinfer-dev_8.5.2-1+cuda11.4_arm64.deb $SYSROOT
dpkg -x $TRT_DIR/libnvinfer-plugin8_8.5.2-1+cuda11.4_arm64.deb $SYSROOT
dpkg -x $TRT_DIR/libnvinfer-plugin-dev_8.5.2-1+cuda11.4_arm64.deb $SYSROOT
dpkg -x $TRT_DIR/libnvonnxparsers8_8.5.2-1+cuda11.4_arm64.deb $SYSROOT
dpkg -x $TRT_DIR/libnvonnxparsers-dev_8.5.2-1+cuda11.4_arm64.deb $SYSROOT
dpkg -x $TRT_DIR/libnvparsers8_8.5.2-1+cuda11.4_arm64.deb $SYSROOT
dpkg -x $TRT_DIR/libnvparsers-dev_8.5.2-1+cuda11.4_arm64.deb $SYSROOT
```

> ⚠️ **注意**：`dpkg -x` 只解壓縮檔案本身，**不會執行套件的 postinst 腳本**。
> 有些檔案（例如 `cudnn.h`）是靠 postinst 用 `update-alternatives` 動態建立的符號連結，
> 不會被展開出來，需要手動補：
>
> ```bash
> ln -s cudnn_v8.h $SYSROOT/usr/include/aarch64-linux-gnu/cudnn.h
> ```

**驗證 sysroot 內容是否完整**：

```bash
ls $SYSROOT/usr/lib/aarch64-linux-gnu/libcudnn*
ls $SYSROOT/usr/lib/aarch64-linux-gnu/libnvinfer*
ls $SYSROOT/usr/include/aarch64-linux-gnu/cudnn_v8.h
ls $SYSROOT/usr/include/aarch64-linux-gnu/NvInfer.h
ls $SYSROOT/usr/include/cudnn.h          # 符號連結
```

---

## 4. 四個核心踩坑紀錄

整體架構：交叉編譯牽涉三套獨立元件互相搭配 ——
**CUDA 11.4 的 nvcc/cicc**、**交叉編譯器（工具鏈）**、**實際的 Jetson sysroot**，
三者對彼此的假設不一致，是所有問題的共同根源。

### 問題一：`__malloc__` does not take arguments

```
/usr/aarch64-linux-gnu/include/stdio.h(189): error: attribute "__malloc__" does not take arguments
```

- **現象**：CUDA 編譯階段，`stdio.h`/`stdlib.h` 大量報錯。
- **根因**：`apt install g++-aarch64-linux-gnu` 裝出來的是 **GCC 11.4**（host 容器是
  Ubuntu 22.04/24.04 的預設版本）。GCC 11 起，glibc 標頭檔的 `__malloc__` 屬性
  改成雙參數寫法，但 **CUDA 11.4 的 `cicc` 前端只支援到 GCC 10**，完全不認得新語法。
- **解法**：改用 NVIDIA 官方為 JetPack 5 / R35 指定的 **Bootlin GCC 9.3** 工具鏈
  （下載頁面：Jetson Linux 下載頁 → TOOLS → **Bootlin Toolchain gcc 9.3**，
  版號 `2020.08-1`）。

  ```cmake
  set(L4T_TOOLCHAIN_BIN /usr/l4t-gcc/bin)
  set(CROSS_TRIPLE aarch64-buildroot-linux-gnu)
  set(CMAKE_C_COMPILER   ${L4T_TOOLCHAIN_BIN}/${CROSS_TRIPLE}-gcc)
  set(CMAKE_CXX_COMPILER ${L4T_TOOLCHAIN_BIN}/${CROSS_TRIPLE}-g++)
  set(CMAKE_CUDA_HOST_COMPILER ${L4T_TOOLCHAIN_BIN}/${CROSS_TRIPLE}-g++)
  ```

### 問題二：`cannot find crt1.o` / `crti.o`

```
ld: cannot find crt1.o: No such file or directory
ld: cannot find crti.o: No such file or directory
```

- **現象**：換上 Bootlin GCC 9.3 後，連結階段找不到 C runtime 啟動檔案。
- **根因**：Bootlin 是 **Buildroot 工具鏈**，預期 `crt1.o` 放在 `${sysroot}/usr/lib/`；
  但 Jetson sysroot 是真正的 Ubuntu 20.04 rootfs，遵循 **Debian multiarch** 慣例，
  實際放在 `${sysroot}/usr/lib/aarch64-linux-gnu/`。編譯器不知道要多找這一層。
- **解法**：加 `-B` 參數明確指定搜尋路徑。

  ```cmake
  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -B${SYSROOT}/usr/lib/aarch64-linux-gnu -B${SYSROOT}/lib/aarch64-linux-gnu")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -B${SYSROOT}/usr/lib/aarch64-linux-gnu -B${SYSROOT}/lib/aarch64-linux-gnu")
  ```

### 問題三：`sys/cdefs.h: No such file or directory`

```
/sysroot/.../usr/include/features.h:461:12: fatal error: sys/cdefs.h: No such file or directory
```

- **現象**：CUDA 編譯（不是連結）階段找不到標頭檔。
- **根因**：跟問題二同源。`sys/cdefs.h` 這類跟架構相關的 glibc 標頭檔，
  一樣被 Ubuntu 放進 `usr/include/aarch64-linux-gnu/` 這個 multiarch 子目錄。
  原生 Debian GCC 因為知道自己的 triple 是 `aarch64-linux-gnu`，會自動搜尋這層；
  Bootlin 的 triple 是 `aarch64-buildroot-linux-gnu`，兜不起來。
- **額外陷阱**：一開始誤用 `CMAKE_INCLUDE_PATH` 想指定這個路徑，
  但這個變數**只有 `find_path()`/`find_file()` 會查詢，不會自動變成編譯器的 `-I` 參數**，
  設定了卻沒有真正生效。
- **解法**：明確加 `-I` 參數。

  ```cmake
  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -I${SYSROOT}/usr/include/aarch64-linux-gnu")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${SYSROOT}/usr/include/aarch64-linux-gnu")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -I${SYSROOT}/usr/include/aarch64-linux-gnu")
  ```

### 問題四：大量 `dlopen`/`pthread_*`/`sem_*` undefined reference

```
undefined reference to `dlopen'
undefined reference to `pthread_create'
undefined reference to `sem_wait'
...
librt.so:undefined reference to `__libc_unwind_link_get@GLIBC_PRIVATE'
```

- **現象**：連結階段一次噴出上百行找不到符號。
- **根因**：CMake 對 CUDA 專案**預設走靜態連結**（`libcudart_static.a`），
  這個靜態版函式庫內部大量呼叫系統函式（`dlopen`、`pthread_create`、`sem_wait` 等），
  正常需要額外連 `-lrt -ldl -lpthread`，但 CMake 內建的試編譯測試不會自動加。
- **解法**：改用動態連結的 `libcudart.so`（Jetson 裝置上本來就有現成的，
  也更貼近實際部署情境）。

  ```cmake
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  ```

---

## 5. 最終可用的 toolchain file

```cmake
# =====================================================
# toolchain-jetson_R3560.cmake
# 用途：Jetson Xavier NX 交叉編譯（含 CUDA aarch64）JetPack 5.1.4 / R35.6.0
# 使用：cmake -DCMAKE_TOOLCHAIN_FILE=toolchain-jetson_R3560.cmake ..
# =====================================================

set(CMAKE_SYSTEM_NAME      Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# --- 交叉編譯器：Bootlin GCC 9.3（不可用 apt 版 GCC 11.4，見問題一）---
set(L4T_TOOLCHAIN_BIN /usr/l4t-gcc/bin)
set(CROSS_TRIPLE aarch64-buildroot-linux-gnu)

set(CMAKE_C_COMPILER   ${L4T_TOOLCHAIN_BIN}/${CROSS_TRIPLE}-gcc)
set(CMAKE_CXX_COMPILER ${L4T_TOOLCHAIN_BIN}/${CROSS_TRIPLE}-g++)

set(CMAKE_AR     ${L4T_TOOLCHAIN_BIN}/${CROSS_TRIPLE}-ar     CACHE FILEPATH "" FORCE)
set(CMAKE_AS     ${L4T_TOOLCHAIN_BIN}/${CROSS_TRIPLE}-as     CACHE FILEPATH "" FORCE)
set(CMAKE_NM     ${L4T_TOOLCHAIN_BIN}/${CROSS_TRIPLE}-nm     CACHE FILEPATH "" FORCE)
set(CMAKE_LINKER ${L4T_TOOLCHAIN_BIN}/${CROSS_TRIPLE}-ld     CACHE FILEPATH "" FORCE)
set(CMAKE_STRIP  ${L4T_TOOLCHAIN_BIN}/${CROSS_TRIPLE}-strip  CACHE FILEPATH "" FORCE)
set(CMAKE_RANLIB ${L4T_TOOLCHAIN_BIN}/${CROSS_TRIPLE}-ranlib CACHE FILEPATH "" FORCE)

# --- Sysroot ---
set(SYSROOT /sysroot/jetson-sysroot_r3560)
set(CMAKE_SYSROOT ${SYSROOT})
set(CMAKE_FIND_ROOT_PATH ${SYSROOT})

# --sysroot + -B（見問題二：crt1.o 搜尋路徑）+ -I（見問題三：標頭檔搜尋路徑）
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} --sysroot=${SYSROOT} -B${SYSROOT}/usr/lib/aarch64-linux-gnu -B${SYSROOT}/lib/aarch64-linux-gnu -I${SYSROOT}/usr/include/aarch64-linux-gnu")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --sysroot=${SYSROOT} -B${SYSROOT}/usr/lib/aarch64-linux-gnu -B${SYSROOT}/lib/aarch64-linux-gnu -I${SYSROOT}/usr/include/aarch64-linux-gnu")

set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} --sysroot=${SYSROOT} -B${SYSROOT}/usr/lib/aarch64-linux-gnu -B${SYSROOT}/lib/aarch64-linux-gnu -Wl,-rpath-link,${SYSROOT}/lib/aarch64-linux-gnu:${SYSROOT}/usr/lib/aarch64-linux-gnu")

set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} -Xcompiler --sysroot=${SYSROOT} -Xlinker --sysroot=${SYSROOT} -Xcompiler -B${SYSROOT}/usr/lib/aarch64-linux-gnu -Xlinker -B${SYSROOT}/usr/lib/aarch64-linux-gnu -Xcompiler -I${SYSROOT}/usr/include/aarch64-linux-gnu")

# --- pkg-config（Bootlin 工具鏈通常不附帶交叉版包裝執行檔）---
set(PKG_CONFIG_EXECUTABLE /usr/bin/pkg-config)
set(ENV{PKG_CONFIG_SYSROOT_DIR} ${SYSROOT})
set(ENV{PKG_CONFIG_LIBDIR}
    "${SYSROOT}/usr/lib/pkgconfig:${SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig")
set(ENV{PKG_CONFIG_ALLOW_SYSTEM_CFLAGS} 1)
set(ENV{PKG_CONFIG_ALLOW_SYSTEM_LIBS}   1)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# --- CUDA Toolkit ---
set(CUDAToolkit_ROOT /usr/local/cuda-11.4)
set(CUDA_TARGET_DIR  ${CUDAToolkit_ROOT}/targets/aarch64-linux)

set(CMAKE_CUDA_COMPILER        ${CUDAToolkit_ROOT}/bin/nvcc)
set(CMAKE_CUDA_COMPILER_TARGET ${CROSS_TRIPLE})
set(CMAKE_CUDA_HOST_COMPILER   ${L4T_TOOLCHAIN_BIN}/${CROSS_TRIPLE}-g++)

# Xavier 系列 = sm_72；Orin 系列則為 sm_87
set(CMAKE_CUDA_ARCHITECTURES 72)

# 動態連結 libcudart.so（見問題四）
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

set(CMAKE_LIBRARY_PATH ${CUDA_TARGET_DIR}/lib ${SYSROOT}/usr/lib)
set(CMAKE_INCLUDE_PATH ${CUDA_TARGET_DIR}/include ${SYSROOT}/usr/include ${SYSROOT}/usr/include/aarch64-linux-gnu)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-I${CUDA_TARGET_DIR}/include>)
add_compile_options($<$<COMPILE_LANGUAGE:C>:-I${CUDA_TARGET_DIR}/include>)
```

**建置指令**：

```bash
rm -rf build
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=/path/to/toolchain-jetson_R3560.cmake ..
make
```

---

## 6. Orin / R36 對照：為什麼同樣做法在新版上更簡單

| 項目 | Xavier / R35（本篇） | Orin / R36 |
|---|---|---|
| CUDA 版本 | 11.4 | 12.6 |
| 交叉編譯器 | Bootlin GCC 9.3（`aarch64-buildroot-linux-gnu`） | `apt` 裝的 `aarch64-linux-gnu-gcc` |
| Sysroot 基底 | Ubuntu 20.04（glibc 2.31） | Ubuntu 22.04（glibc 2.35） |

- **問題一不會發生**：CUDA 12.6 的 `cicc` 官方支援到 GCC 12/13，`apt` 裝的較新版
  編譯器不會觸發 `__malloc__` 語法錯誤。
- **問題二、三不會發生**：`aarch64-linux-gnu` 是 Debian 官方認證的 triple，
  `apt` 裝出來的編譯器本身就知道要自動搜尋 multiarch 子目錄，不需要手動加 `-B`/`-I`。
- **問題四不會發生**：glibc 2.34（2021 年 8 月）起，`libpthread`/`librt`/`libdl`
  已合併進 `libc.so.6` 本體。Ubuntu 22.04 的 glibc 2.35 已過這次合併，
  即使靜態連結 `libcudart_static.a` 也不會缺 `pthread_*`/`dlopen` 等符號。

**結論**：這幾個坑本質上都是 **CUDA 11.4 世代被迫搭配「較舊編譯器 + 較舊 glibc」**
所產生的相容性摩擦，換到 CUDA 12.x + Ubuntu 22.04 的世代組合後就自然消失。
