1. set environment
    in docker need add option --privileged
    - apt-get install gawk wget git diffstat unzip texinfo gcc build-essential chrpath socat cpio python3 python3-pip python3-pexpect xz-utils debianutils iputils-ping python3-git python3-jinja2 libegl1-mesa libelf-dev libsdl1.2-dev lz4 pylint xterm python3-subunit mesa-common-dev curl udev
    - /lib/systemd/systemd-udevd --daemon
    - usermod -aG dialout $(whoami)
    - mkdir -p ~/bin
    - PATH="${HOME}/bin:${PATH}"
    - curl https://storage.googleapis.com/git-repo-downloads/repo > ~/bin/repo
    - chmod a+rx ~/bin/repo
    - apt-get install android-tools-adb android-tools-fastboot
    - echo -n 'SUBSYSTEM=="usb", ATTR{idVendor}=="0e8d", ATTR{idProduct}=="201c", MODE="0660", TAG+="uaccess"
SUBSYSTEM=="usb", ATTR{idVendor}=="0e8d", ATTR{idProduct}=="0003", MODE="0660", TAG+="uaccess"
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", MODE="0660", TAG+="uaccess"
SUBSYSTEM=="gpio", MODE="0660", TAG+="uaccess"
' | sudo tee /etc/udev/rules.d/72-aiot.rules
    - sudo udevadm control --reload-rules
    - sudo udevadm trigger
    - apt-get install picocom
    - pip3 install -U genio-tools
    - apt install gcc-arm-linux-gnueabihf
