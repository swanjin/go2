# unitree_sdk2
Unitree robot sdk version 2.

### Prebuild environment
* OS  (Ubuntu 20.04 LTS)  
* CPU  (aarch64 and x86_64)   
* Compiler  (gcc version 9.4.0) 

### Installation
```bash
sudo ./install.sh

```

### Build examples
```bash
mkdir build
cd build
cmake ..
make
```

If you want to build the python wrapper, then replace the cmake line with:
```bash
cmake -DPYTHON_BUILD=TRUE ..
```

### Notice
For more reference information, please go to [Unitree Document Center](https://support.unitree.com/home/zh/developer).
