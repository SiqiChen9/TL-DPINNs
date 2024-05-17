# Causality-enhanced Discreted Physics-informed Neural Networks for Predicting Evolutionary Equations

## Abstract

> Physics-informed neural networks (PINNs) have shown promising potential for solving partial differential equations (PDEs) using deep learning. However, PINNs face training difficulties for evolutionary PDEs, particularly for dynamical systems whose solutions exhibit multi-scale or turbulent behavior over time. The reason is that PINNs may violate the temporal causality property since all the temporal features in the PINNs loss are trained simultaneously. This paper proposes to use implicit time differencing schemes to enforce temporal causality, and use transfer learning to sequentially update the PINNs in space as surrogates for PDE solutions in different time frames. The evolving PINNs are better able to capture the varying complexities of the evolutionary equations, while only requiring minor updates between adjacent time frames. Our method is theoretically proven to be convergent if the time step is small and each PINN in different time frames is well-trained. In addition, we provide state-of-the-art (SOTA) numerical results for a variety of benchmarks for which existing PINNs formulations may fail or be inefficient. We demonstrate that the proposed method improves the accuracy of PINNs approximation for evolutionary PDEs and improves efficiency by a factor of 4–40x.


## Examples

### Reaction-Diffusion equation
Contrast  
![image](https://user-images.githubusercontent.com/88814995/235424556-3d458e5d-5a04-4865-996d-d52a281a9ff3.png)  

Time stamps  
![image](https://user-images.githubusercontent.com/88814995/235424580-1c3e9d9c-915e-4e93-9076-3b505c5fee4d.png)  

### Allen-Cahn equation
Contrast  
![image](https://user-images.githubusercontent.com/88814995/235424754-cf30d853-7d53-4203-8da8-a63c8f66183b.png)  

Time stamps  
![image](https://user-images.githubusercontent.com/88814995/235424768-adafd109-f569-4a7b-8332-525915c49fa8.png)  

### Kuramoto–Sivashinsky (regular) equation
Contrast  
![image](https://user-images.githubusercontent.com/88814995/235425044-3ff79386-6a61-4f87-b566-4d5149ce91a3.png)  

Time stamps  
![image](https://user-images.githubusercontent.com/88814995/235425105-2b1e26bf-90d4-4cf1-9c8d-b11379b7ef8c.png)  

https://user-images.githubusercontent.com/88814995/235442682-63990632-713a-4418-8da7-7af5dfa22fa9.mp4

### Kuramoto–Sivashinsky (chaotic) equation
Contrast  
![image](https://user-images.githubusercontent.com/88814995/235425374-3ca7eb30-4fe2-4c43-8a94-f6b528fde834.png)  

Time stamps  
![image](https://user-images.githubusercontent.com/88814995/235425395-fdad9fb4-04f5-4092-ac9f-a19eac5a1403.png)  

https://user-images.githubusercontent.com/88814995/235442726-a70b77c2-963a-40d2-ba8c-ccf7cc035810.mp4

### Navier-Stokes equation
Relative $L^2$ error each time stamp  
![image](https://user-images.githubusercontent.com/88814995/235425516-668e6179-1f52-4590-a646-2144eb604dff.png)  

https://user-images.githubusercontent.com/88814995/235442765-a7c5d18e-26ab-4b55-a5a1-d0a6fd8e5c99.mp4
