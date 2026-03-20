在 Python 3.8 之後引入的 `multiprocessing.shared_memory` 模組，正是為了解決多進程（Multi-processing）間傳遞大量數據時，因序列化（Pickle）導致的效能瓶頸。

### 什麼是 SharedMemory？

簡單來說，`SharedMemory` 允許不同的進程直接存取同一塊**物理記憶體區域**。

- **傳統方式**：進程 A 將資料透過 Queue 或 Pipe 傳給進程 B，中間必須經過「序列化 → 傳輸 → 反序列化」，這在資料量大（如數 GB 的 NumPy Array）時非常慢。
    
- **共享記憶體**：進程 A 把資料寫入記憶體，進程 B 直接讀取同一塊位址，實現 **Zero-copy** 的效率。
    

---

### 核心操作流程

操作 `SharedMemory` 通常分為「創建端」與「接入端」：

#### 1. 創建端 (Creator)

負責配置記憶體大小並賦予一個唯一的名稱。

``` python
from multiprocessing import shared_memory

# 建立一塊 1024 位元組的共享記憶體
shm_a = shared_memory.SharedMemory(name="my_shared_data", create=True, size=1024)

# 寫入資料 (必須是 bytes-like object)
buffer = shm_a.buf
buffer[:11] = b"Hello World"

# 先不要 close，否則接入端會找不到
```

#### 2. 接入端 (Consumer)

透過名稱（name）直接掛載該記憶體。

```python
from multiprocessing import shared_memory

# 接入已存在的共享記憶體
shm_b = shared_memory.SharedMemory(name="my_shared_data")

print(bytes(shm_b.buf[:11]))  # 輸出: b'Hello World'

# 使用完畢後釋放連結
shm_b.close()
```

### 與 NumPy 結合

在機器學習任務中，我們常需要多個 Worker 同時處理同一個大型矩陣。透過 `SharedMemory` 結合 `np.ndarray` 的 `buffer` 參數，可以達成極速存取：

``` python
import numpy as np
from multiprocessing import shared_memory

# 1. 建立原始資料
data = np.array([1, 2, 3, 4, 5], dtype=np.int64)

# 2. 建立共享記憶體
shm = shared_memory.SharedMemory(create=True, size=data.nbytes)

# 3. 在共享記憶體上建立一個 NumPy 視圖 (View)
shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)

# 4. 將資料拷貝進去
shared_array[:] = data[:]

print(f"共享記憶體名稱: {shm.name}")
```


### 注意事項與陷阱

- **生命週期管理**：
    
    - `close()`：僅代表當前進程不再使用這塊記憶體。
        
    - `unlink()`：**徹底銷毀**這塊記憶體。通常由「創建端」在確定所有進程都結束後呼叫，否則會造成記憶體洩漏（Memory Leak）。
        
- **同步問題**：`SharedMemory` 本身**不具備鎖（Lock）機制**。如果多個進程同時寫入，會發生 Race Condition，建議搭配 `multiprocessing.Lock` 使用。
    
- **平台差異**：在 Windows 上，如果所有指向該記憶體的進程都關閉了，系統可能會自動回收；在 Linux/Unix 上則會持續存在直到呼叫 `unlink()` 或重啟系統。