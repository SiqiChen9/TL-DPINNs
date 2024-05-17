# Causality-enhanced Discreted Physics-informed Neural Networks for Predicting Evolutionary Equations

## Abstract

> Physics-informed neural networks (PINNs) have shown promising potential for solving partial differential equations (PDEs) using deep learning. However, PINNs face training difficulties for evolutionary PDEs, particularly for dynamical systems whose solutions exhibit multi-scale or turbulent behavior over time. The reason is that PINNs may violate the temporal causality property since all the temporal features in the PINNs loss are trained simultaneously. This paper proposes to use implicit time differencing schemes to enforce temporal causality, and use transfer learning to sequentially update the PINNs in space as surrogates for PDE solutions in different time frames. The evolving PINNs are better able to capture the varying complexities of the evolutionary equations, while only requiring minor updates between adjacent time frames. Our method is theoretically proven to be convergent if the time step is small and each PINN in different time frames is well-trained. In addition, we provide state-of-the-art (SOTA) numerical results for a variety of benchmarks for which existing PINNs formulations may fail or be inefficient. We demonstrate that the proposed method improves the accuracy of PINNs approximation for evolutionary PDEs and improves efficiency by a factor of 4–40x.


## Examples

### Reaction-Diffusion equation
![contrast_RD](https://github.com/SiqiChen9/TL-DPINNs/assets/133206108/1449aeaf-3afb-410a-8c3b-52833146390e) 

### Allen-Cahn equation
<img width="1109" alt="contrast_AC_eg" src="https://github.com/SiqiChen9/TL-DPINNs/assets/133206108/64cf6ae7-4c4e-4ca2-876b-a157763dfae8">

### Kuramoto–Sivashinsky (regular) equation
![contrast_KSr](https://github.com/SiqiChen9/TL-DPINNs/assets/133206108/2eeaed19-3be9-4be5-9f9b-b5bc5ccea8ed)


https://github.com/SiqiChen9/TL-DPINNs/assets/133206108/ac3212e1-05d7-4a16-973b-6ced930fc59b


### Kuramoto–Sivashinsky (chaotic) equation
![contrast_KSc](https://github.com/SiqiChen9/TL-DPINNs/assets/133206108/d326265b-63a9-4338-a2b3-698286504161)


https://github.com/SiqiChen9/TL-DPINNs/assets/133206108/ee78d8aa-5aac-4123-9826-895b63622459


### Navier-Stokes equation
![L2_NS](https://github.com/SiqiChen9/TL-DPINNs/assets/133206108/b9e5e6f1-9453-4780-be48-4d73d25f2475)


https://github.com/SiqiChen9/TL-DPINNs/assets/133206108/d262f0d7-6eb8-4c24-9911-3f19b4fcb2be

