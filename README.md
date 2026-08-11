

<div align="center">

# APAN: Adaptive Prototype-Aware Attention Network for Aerial-Ground Person Re-Identification



[Xianxian Zeng](), 
[Yijun Chen](https://github.com/ChenYiJun630), 
[Jun Yuan](), 
[Jiawen Li](), 
[Rongjun Chen](), 
[Huimin Zhao](), 
[Jinchang Ren](), 
[Shun Liu*]()


<!-- [![Paper](https://img.shields.io/badge/Paper-arXiv%3A2504.09549-b31b1b.svg)]()
[![PDF](https://img.shields.io/badge/PDF-arXiv-2b6cb0.svg)]()
[![Venue](https://img.shields.io/badge/Venue-IEEE%20TMM%202026-111827.svg)](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=83) -->
[![Python](https://img.shields.io/badge/Python-3.11-3776ab.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Apach-red.svg)](LICENSE)


</div>

## 📝 Abstract
Aerial-Ground Person Re-identification (AGPReID), which aims to match pedestrians across UAV and ground cameras, represents a critical multi-modal retrieval task at the intersection of vision and multimedia analysis. While effective in homogeneous settings, conventional person Re-identification (ReID) models falter under the extreme viewpoint and domain disparity inherent to AGPReID, rendering global features inadequate. Current methods seeking view-invariant representations still face difficulties in capturing robust local features, hampered by large intra-class variations and subtle inter-class differences. To address this, we propose an Adaptive Prototype-aware Attention Network (APAN) that learns view-invariant local features without auxiliary supervision. APAN operates through two synergistic modules: a Prototype Relation Modeling (PRM) module that establishes dynamic semantic centers and token-prototype associations, and a Prototype Information Flow Module (PIFM) that refines and propagates prototype-enhanced features in a closed loop. Furthermore, we introduce an Adaptive Distributional Margin Loss, which employs a mode-seeking mechanism to sharpen intra-class distributions and widen inter-class margins. Extensive experiments on four benchmarks validate that our approach achieves state-of-the-art performance.

![PRM](./assets/PRM.jpg)
![PIFM](./assets/PIFM.jpg)


## 🧠 Method

### 📦 Requirements
#### Step1: Prepare environments

Please refer to [INSTALL.md](./INSTALL.md).

#### Step2: Prepare datasets
- **CARGO:**   [Download here](https://drive.google.com/file/d/1yDjyH0VtW7efxP3vgQjIqTx2oafCB67t/view?usp=drive_link)
- **AG-ReID.v2:**  [Download here](https://drive.google.com/drive/folders/16r7G_CuUqfWG6_UCT7goIGRMqJird6vK?usp=share_link)
- **AG-ReID:**  [Download here](https://drive.google.com/file/d/1hzieEPlXfjkN3V3XWqI5rAwpF_sCF1K9/view?usp=sharing)
- **LAGPeR:**  [Download here](https://github.com/wangshining681/SeCap-AGPReID/tree/main)

Download the datasets and modify the dataset path.  

#### Step3: Prepare ViT Pre-trained Models

Download the ViT-base Pre-trained model and modify the path in [configs](./configs/CARGO/apan.yml):

> PRETRAIN_PATH: xxx

### 🚀 Training & Testing


Use the following command to train APAN. We use 4 × RTX 3090 GPUs for training; you can also train with a single GPU, but it will be much slower:

```bash
bash train.sh
```


Use the following command to evaluate APAN. The trained weights for testing are available at [Google Drive](https://drive.google.com/drive/my-drive?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto).

```bash
bash test.sh
```





## 📊 Results

**Comparative retrieval performance on the CARGO dataset.** The highest and second-highest scores are formatted in bold and underlined text, respectively.
![CARGO](./assets/CARGO.png)

**Comparative retrieval performance on the AG-ReID.v2 dataset.** The highest and second-highest scores are formatted in bold and underlined text, respectively.

<div align="center"><img src="./assets/AGReIDv2.png" width="60%"></div>


**Comparative retrieval performance on the AG-ReID dataset.** The highest and second-highest scores are formatted in bold and underlined text, respectively.

<div align="center"><img src="./assets/AGReID.png" width="60%"></div>

**Comparative retrieval performance on the LAGPeR dataset.** The highest and second-highest scores are formatted in bold and underlined text, respectively.

<div align="center"><img src="./assets/LAGPeR.png" width="60%"></div>

## 🙏 Acknowledgement
Codebase from [VDT](https://github.com/LinlyAC/VDT-AGPReID) and [SeCap](https://github.com/wangshining681/SeCap-AGPReID).
We sincerely thank the authors and contributors of these projects for their valuable contributions.

## 📖 Citation
If you find APAN useful in your research, please consider citing: