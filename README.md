# Hyperspectral Image Classification

This repository implementates 6 frameworks for hyperspectral image classification based on PyTorch and sklearn.

The detailed results can be seen in the [Classification of Hyperspectral Image Based on 
Double-Branch Dual-Attention Mechanism Network ]().

Feel free to contact me if you need any further information: lironui@whu.edu.cn

Some of our code references the projects
* [Remote sensing image classification](https://github.com/stop68/Remote-Sensing-Image-Classification.git)
* [Implementation of SSRN for Hyperspectral Image Classification](https://github.com/zilongzhong/SSRN.git)
* [A Fast Dense Spectral-Spatial Convolution Network Framework for Hyperspectral Images Classification](https://github.com/shuguang-52/FDSSC.git) 

Requirements：
------- 
```
numpy >= 1.16.5
PyTorch >= 1.3.1
sklearn >= 0.20.4
```

Datasets:
------- 
You can download the hyperspectral datasets in mat format at: [](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes), and move the files to `./datasets` folder.

Usage:
------- 
1. Set the percentage of training and validation samples by the `load_dataset` function in the file `./global_module/generate_pic.py`.
2. Taking the DBDA framework as an example, run `./DBDA/main.py` and type the name of dataset. 
3. The classfication maps are obtained in `./DBDA/classification_maps` folder, and accuracy result is generated in `./DBDA/records` folder.

Network:
------- 
* [DBDA]()
* [DBMA](https://www.mdpi.com/2072-4292/11/11/1307/xml)
* [FDSSC](https://www.mdpi.com/2072-4292/10/7/1068/htm)
* [SSRN](https://ieeexplore.ieee.org/document/8061020)
* [CDCNN](https://ieeexplore.ieee.org/document/7973178)
* [SVM](https://ieeexplore.ieee.org/document/1323134/)

![network](https://github.com/lironui/Double-Branch-Dual-Attention-Mechanism-Network/figure/figures/Figure 6. The structure of the DBDA network.png)  
Figure 6. The structure of our DBDA network.
