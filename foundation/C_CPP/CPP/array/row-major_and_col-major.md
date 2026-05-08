## 1️⃣ Row-major（行優先）

- **C / C++** 預設使用 row-major。
    
- **規則**：矩陣的每一行（row）依次排列在記憶體中。
    
- **記憶體排列**：先把第 0 行放下，再第 1 行，再第 2 行…

``` cpp
int a[3][4] = { 
    {0,1,2,3},
    {4,5,6,7},
    {8,9,10,11}
};
```

記憶體（row-major）排布：

``` bash
[0][1][2][3][4][5][6][7][8][9][10][11]
```

映射到二維：

``` bash
Row 0: 0  1  2  3
Row 1: 4  5  6  7
Row 2: 8  9 10 11
```

**公式**：

``` bash
對應一維地址: a[i][j] = base_address + (i * num_cols + j)
```

## 2️⃣ Column-major（列優先）

- **Fortran / MATLAB / Julia** 預設使用 column-major。
    
- **規則**：矩陣的每一列（column）依次排列在記憶體中。
    
- **記憶體排列**：先把第 0 列放下，再第 1 列，再第 2 列…
    

以同樣矩陣為例，如果用 column-major：

``` bash
[0][4][8][1][5][9][2][6][10][3][7][11]
```

映射到二維：

``` bash 
ol 0: 0  4  8 Col 1: 1  5  9 Col 2: 2  6 10 Col 3: 3  7 11
```

**公式**：

``` bash
對應一維地址: a[i][j] = base_address + (j * num_rows + i)
```

---

## 3️⃣ 對比重點

||Row-major (C/C++)|Column-major (Fortran/MATLAB)|
|---|---|---|
|儲存順序|行 → 行 → 行|列 → 列 → 列|
|訪問 a[i][j]|base + (i * num_cols + j)|base + (j * num_rows + i)|
|典型語言|C, C++|Fortran, MATLAB, Julia|
|初始化方式|`{ {row1}, {row2}, … }` 或扁平化|`{value按列順序排列}`|

---

💡 **直覺**：

- row-major：每行連續 → 水平掃描更快。
    
- column-major：每列連續 → 垂直掃描更快。