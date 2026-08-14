## 1️⃣ 靜態二維陣列 + 指標操作

假設有一個 3x3 矩陣：

``` cpp
#include <iostream>
using namespace std;

int main() {
    int mat[3][3] = {
        {1,2,3},
        {4,5,6},
        {7,8,9}
    };

    int (*p)[3] = mat;  // p 是指向長度為3陣列的指標

    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) {
            cout << *(*(p + i) + j) << " ";  // 指標算術: p+i 指向第 i 列
        }
        cout << endl;
    }

    return 0;
}
```

✅ 重點：

- `int (*p)[3]` 表示「指向長度為 3 的整數陣列的指標」。
    
- `*(p+i)` 取得第 i 列，`*(*(p+i)+j)` 取得第 i 列第 j 欄的值。
    
- 對於靜態陣列，這種方式安全且簡單。
    

---

## 2️⃣ 動態二維陣列 + 指標操作

如果矩陣大小在編譯時不確定，就需要動態配置：

``` cpp
#include <iostream>
using namespace std;

int main() {
    int rows = 3, cols = 3;

    // 配置二維矩陣
    int **mat = new int*[rows];
    for(int i=0; i<rows; i++) {
        mat[i] = new int[cols];
    }

    // 初始化矩陣
    for(int i=0; i<rows; i++)
        for(int j=0; j<cols; j++)
            mat[i][j] = i * cols + j + 1;

    // 用指標操作
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            cout << *(*(mat + i) + j) << " ";
        }
        cout << endl;
    }

    // 釋放記憶體
    for(int i=0; i<rows; i++)
        delete[] mat[i];
    delete[] mat;

    return 0;
}
```

✅ 重點：

- `int **mat` 是指向指標陣列的指標，每一列都是動態配置的。
    
- 訪問用法仍是 `*(*(mat+i)+j)`。
    
- 釋放記憶體時一定要**先釋放列，再釋放 mat**。



---

## 3️⃣ 一維陣列模擬二維矩陣（常用於 GPU / 效能敏感場合）

``` cpp
#include <iostream>
using namespace std;

int main() {
    int rows = 3, cols = 3;
    int *mat = new int[rows*cols];

    // 初始化
    for(int i=0; i<rows; i++)
        for(int j=0; j<cols; j++)
            *(mat + i*cols + j) = i*cols + j + 1;

    // 訪問
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            cout << *(mat + i*cols + j) << " ";
        }
        cout << endl;
    }

    delete[] mat;
    return 0;
}
```

✅ 重點：

- 矩陣只用一塊連續記憶體，訪問時用 `i*cols + j` 計算偏移。
    
- 更接近 CUDA / GPU 的做法，cache 效能好。
    

---

💡 **小技巧**：

1. 指標算術比 `mat[i][j]` 更靈活，但容易出錯。
    
2. 若矩陣很大或多維，建議用一維陣列 + 自行計算偏移。
    
3. 靜態陣列可以用 `int (*p)[cols]`，動態陣列建議用一維陣列或 `std::vector`。