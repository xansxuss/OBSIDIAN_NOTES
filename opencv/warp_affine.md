## **Warp Affine (仿射變換)**

- **特性**：這是一種「二維線性變換」加上「平移」。它最大的特色是 **「維持平行性」**。
    
- **視覺表現**：如果原圖中有兩條平行線，變換後它們**依然平行**。它只能進行旋轉、平移、縮放和剪切（Shear）。
    
- **自由度 (DOF = 6)**：矩陣中有 6 個未知數。你只需要 **3 個點**（構成一個三角形）在變換前後的對應關係，就能解出這個矩陣。

## 數學模型深究

在工程實作（如 OpenCV）中，兩者的計算方式如下：

## **仿射變換矩陣 ($2 \times 3$)**

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} a_{11} & a_{12} & t_x \\ a_{21} & a_{22} & t_y \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

這本質上是在 2D 平面上的旋轉、縮放與平移。

## 常見的仿射變換類型

仿射變換是多種幾何操作的集合體：

- **平移 (Translation):** 改變 $b_0$ 與 $b_1$。
    
- **縮放 (Scaling):** 改變對角線元素 $a_{00}$ 與 $a_{11}$。
    
- **旋轉 (Rotation):** 結合三角函數 $\sin$ 與 $\cos$ 修改 $a_{ij}$。
    
- **剪切 (Shearing):** 修改非對角線元素 $a_{01}$ 或 $a_{10}$。

## 應用場景

## **什麼時候用 Warp Affine？**

1. **Face Alignment (人臉對齊)**：當你偵測到人臉的五官關鍵點（Landmarks），通常會用 Affine 將雙眼對齊到水平線上，因為人臉在照片中通常被視為平面。
    
2. **Data Augmentation (資料增強)**：基本的旋轉、平移、縮放。
    
3. **OCR 預處理**：當文字稍微傾斜時，用仿射變換進行水平校正。

## OpenCV 函數快速對照

| **功能**    | **Warp Affine**                      | **Warp Perspective**                      |
| --------- | ------------------------------------ | ----------------------------------------- |
| **求變換矩陣** | `cv2.getAffineTransform(pts1, pts2)` | `cv2.getPerspectiveTransform(pts1, pts2)` |
| **執行變換**  | `cv2.warpAffine(src, M, dsize)`      | `cv2.warpPerspective(src, M, dsize)`      |

如果一堆特徵點（例如 SIFT/ORB 匹配出來的點對），通常會搭配 `cv2.findHomography()` 來計算透視矩陣，並加上 RANSAC 演算法來排除離群點（Outliers）。