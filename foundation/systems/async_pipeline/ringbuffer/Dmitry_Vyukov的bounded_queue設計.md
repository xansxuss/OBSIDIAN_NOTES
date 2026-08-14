Dmitry Vyukov 的 MPMC bounded queue（常被稱為 **Vyukov queue** 或 **MPMC bounded queue**）核心巧思是：**幫每一格（cell）配一個獨立的 sequence number**，把「搶格子」跟「格子資料真正寫完」這兩件事解耦，藉此避免「naive CAS head/tail」資料還沒寫完就被消費者讀到的問題。

### 演算法核心概念

先講直覺：naive 版本用單一 `head`/`tail` CAS 的問題是——生產者 A 用 CAS 把 `head` 從 5 搶到 6（代表「第 5 格是我的了」），但**還沒開始寫資料**，這時候如果 `tail` 也追上來到 5，消費者就可能讀到一個空的或舊的格子。

Vyukov 的解法：每格自己帶一個 `sequence`，初始值就是它的索引。生產者/消費者搶格子時，不只看 `head`/`tail`，還要看**這一格現在轉到第幾圈（lap）**，用這個來判斷格子到底「可以寫」還是「可以讀」。

#### 資料結構

cpp

```cpp
struct Cell {
    std::atomic<size_t> sequence;
    T data;
};

Cell buffer[SIZE];              // SIZE 需為 2 的冪次
std::atomic<size_t> enqueue_pos; // 生產者搶號用
std::atomic<size_t> dequeue_pos; // 消費者搶號用
```

初始化時 `buffer[i].sequence = i`。

#### Enqueue（生產者）

```
pos = enqueue_pos.load(relaxed)
loop:
    cell = buffer[pos & MASK]
    seq  = cell.sequence.load(acquire)
    diff = seq - pos

    if diff == 0:
        # 這格輪到「第 pos 次使用」，現在可以寫
        if CAS(enqueue_pos, pos, pos+1):   # 搶到這個位置
            break
        # 沒搶到，pos 被其他 CAS 改了，reload 重試
    elif diff < 0:
        return FULL   # 這格還沒被消費者釋放，佇列滿了
    else:
        pos = enqueue_pos.load(relaxed)  # 別人搶走了，重讀最新值再試

cell.data = item
cell.sequence.store(pos + 1, release)   # 「發佈」：告訴消費者這格可以讀了
```

#### Dequeue（消費者）

```
pos = dequeue_pos.load(relaxed)
loop:
    cell = buffer[pos & MASK]
    seq  = cell.sequence.load(acquire)
    diff = seq - (pos + 1)

    if diff == 0:
        # 生產者已經發佈完成，可以讀
        if CAS(dequeue_pos, pos, pos+1):
            break
    elif diff < 0:
        return EMPTY
    else:
        pos = dequeue_pos.load(relaxed)

item = move(cell.data)
cell.sequence.store(pos + SIZE, release)  # 標記這格「下一圈」可再被生產者用
```

### 為什麼這樣就正確了

1. **CAS 只搶「位置」，不搶「格子資料」**：`enqueue_pos`/`dequeue_pos` 上的 CAS 保證同一個 `pos` 只有一個生產者（或消費者）搶得到，所以不會有兩個執行緒同時寫同一格
2. **`sequence` 就是格子的「號誌燈」**：生產者寫完資料才把 `sequence` 設成 `pos+1`（release），消費者用 `acquire` 讀 `sequence`，這組 acquire/release 配對確保**消費者一定是在資料真正寫完之後才讀得到**
3. **沒有 ABA 問題**：`sequence` 只會往上加，同一格「第 N 次使用」的號碼在邏輯上是唯一的，不會像單純的指標重複使用那樣被誤判