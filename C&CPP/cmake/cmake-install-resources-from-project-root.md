### 在 CMake 中，若要將專案主目錄（Source Directory）下的資料夾複製到安裝位置（Install Prefix），最標準且推薦的做法是使用 install(DIRECTORY ...) 指令。

#### 1. 使用 install(DIRECTORY) (最推薦)
這是最符合 CMake 規範的做法，會在執行 make install 或 cmake --install . 時觸發。CMake# 假設要將主目錄下的 "configs" 資料夾複製到安裝目錄下

``` CMake
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/configs
        DESTINATION . # 相對於 CMAKE_INSTALL_PREFIX 的路徑
        FILES_MATCHING 
        PATTERN "*.json"     # 可以篩選特定檔案類型
        PATTERN ".git" EXCLUDE # 排除版本控制資料夾
)
```

重點筆記：

- 結尾斜線的差異：
    - configs (無斜線)：會在目標路徑建立 configs 資料夾，結果為 bin/configs/...。
    - configs/ (有斜線)：只會複製資料夾內的內容到目標路徑。
- DESTINATION： 通常會設為 share/${PROJECT_NAME} 或直接點 .（即安裝根目錄）。

#### 2. 在編譯時期複製 (Build Time)如果你希望在編譯過程中（尚未執行 install 前）就在 Build 目錄看到這些資料夾（例如為了執行單元測試），可以使用 file(COPY ...)：

``` CMake
(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)
```

注意： file(COPY) 是在 CMake 設定階段 (Configure time) 執行。如果資料夾內的檔案有變動，除非重新 run CMake，否則不會自動更新。

#### 3. 使用 add_custom_command (動態更新)如果你希望每次 make 的時候，若資料夾有變動就自動同步，這在開發階段很有用：CMakeadd_custom_command(

``` CMake
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/assets
        ${CMAKE_CURRENT_BINARY_DIR}/assets
    COMMENT "Copying assets to build directory..."
)
```

#### 總結建議表格:

| 需求場景 | 推薦指令 | 執行時機 |
| --- | --- | --- |
| 最終打包/佈署 | install(DIRECTORY ...) | 執行 Install 時 |
| 簡單一次性複製 | file(COPY ...) | CMake 設定階段 |
| 開發中自動同步 | add_custom_command | 編譯完成後 (Post-build) |