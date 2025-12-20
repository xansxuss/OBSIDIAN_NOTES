pip install tensorflow==2.6.0
### test
```
import tensorflow as tf
print(tf.test.is_gpu_available())
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, '\n', cpus)
```

python3 -c "import tensorflow as tf; print(tf.test.is_gpu_available())"
python3 -c "import tensorflow as tf; gpus = tf.config.experimental.list_physical_devices(device_type='GPU'); print(gpus, '\n')"