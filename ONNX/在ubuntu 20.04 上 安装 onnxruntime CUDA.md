[在ubuntu 20.04 上 安装 onnxruntime CUDA](https://blog.csdn.net/qq_42995327/article/details/121034487)
[Ubuntu下的onnxruntime(c++)install](https://blog.51cto.com/u_15699099/5649211)
[cuda 12 環境下 onnxruntime-gpu 安装](https://hackmd.io/@iii-gai/Byjrqrb00?utm_source=preview-mode&utm_medium=rec)
import onnxruntime as ort
print(ort.get_available_providers())

``` python
import onnxruntime as ort

# 建立一個超小的虛擬 Session 測試
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
try:
    # 這裡我們不載入模型，只初始化 Provider
    opts = ort.SessionOptions()
    sess = ort.InferenceSession(bytes(), opts, providers=providers)
except Exception as e:
    # 這裡通常會噴出真正的 libcudnn.so.x 找不到的錯誤
    print(f"深度錯誤資訊: {e}")

print(f"實際使用的 Provider: {ort.get_available_providers()}")
```