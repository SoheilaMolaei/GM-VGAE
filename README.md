# GM-VGAE
This repository contains a PyTorch implementation of our paper, **Graph representation learning based on deep generative gaussian mixture models** for graph representation Learning. Here, we propose the Gaussian Mixture Model (GMM) to model the prior distribution in VGAE. Furthermore, an adversarial regularization is incorporated into the proposed approach to ensure the fruitful impact of the latent representations on the results.
<img width="859" alt="image" src="https://github.com/SoheilaMolaei/GM-VGAE/assets/63698187/6ca4eeaa-8ff6-46e6-a64a-b715ea77d945">
# Requirements
- Pytorch
  - !pip install torch-scatter
  - !pip install torch-sparse
  - !pip install torch-cluster
  - !pip install torch-spline-conv 
  - !pip install torch-geometric==1.0.2
  - !pip install torchvision
- python 3.x
- networkx
- scikit-learn
- scipy

- # Cite
Please cite our paper if you use this code in your own work:
```
@article{niknam2023graph,
  title={Graph representation learning based on deep generative gaussian mixture models},
  author={Niknam, Ghazaleh and Molaei, Soheila and Zare, Hadi and Clifton, David and Pan, Shirui},
  journal={Neurocomputing},
  volume={523},
  pages={157--169},
  year={2023},
  publisher={Elsevier}
}
```
