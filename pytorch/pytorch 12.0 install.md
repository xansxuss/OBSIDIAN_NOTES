install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
### test
``` 
import torch
print(torch.cuda.is_available())
```

python3 -c "import torch; print(torch.cuda.is_available())"

### pytorch docker image
https://hub.docker.com/r/pytorch/pytorch/tags?page=1