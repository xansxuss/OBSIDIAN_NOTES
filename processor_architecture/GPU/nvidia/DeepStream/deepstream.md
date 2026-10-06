deepstreamer
install
    1. [install doc](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html#dgpu-setup-for-ubuntu)
use 
    1. [start Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html)

not find libucs.so((gst-plugin-scanner:224167): GStreamer-WARNING **: 10:36:27.291: Failed to load plugin '/usr/lib/x86_64-linux-gnu/gstreamer-1.0/deepstream/libnvdsgst_ucx.so': libucs.so.0: cannot open shared object file: No such file or directory
)
sudo apt-get install libucx0 libucx-dev