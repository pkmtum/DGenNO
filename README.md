# Deep Generative Neural Operator 
The implementation of the [Deep Generative Neural Operator (DGenNO)](https://arxiv.org/pdf/2502.06250).

**Note:** The implementation of Deep Generative Neural Operator (DGenNO) and other popular deep neural operator (DNO) methods (e.g., DeepONet, FNO, PI-DeepONet, and PINO) can also be found on the Github repository: [Deep-Neural-Operators-for-PDEs](https://github.com/yaohua32/Deep-Neural-Operators-for-PDEs).

### (1) The DGenNO framework (<p><span style="color: red;">without any labeled input-output training pairs</span><p>)
<p align="center">
  <img src="./Docs/DGM4DGNO.png" alt="DGenNO" width="800" height='300'/>
</p>
<p align="center">
  <em>Figure 1: The Deep Generative Neural Operator (DGenNO) framework.</em>
</p>

### (2) DGenNO for inverse problems with <p><span style="color: red;">noisy, sparse observations</span><p>
<p align="center">
  <img src="./Docs/DGNO4Inverse.png" alt="DGenNO4Inverse" width="800" height='300'/>
</p>
<p align="center">
  <em>Figure 2: Solving the inverse problem with DGenNO.</em>
</p>

### (3) The MultiONet architecture
<p align="center">
  <img src="./Docs/MultiONet.png" alt="MultiONet" width="800" height='600'/>
</p>
<p align="center">
  <em>Figure 3: a) DeepONet architecture vs. b) MultiONet architecture.</em>
</p>

## 📌 Benchmark Problems
We evaluate the DNO frameworks on the following PDEs:
#### 1. Burger’s Equation
Goal: Learn the operator mapping initial condition $a(x):=u(x,t=0)$ to the solution $u(x,t)$.

#### 2. Darcy’s Flow
Goal: Learn the mapping from the permeability field $a(x)$ to the pressure field $u(x)$.
We considered two cases: (1) Smooth $a(x)$ and (2) Piecewise-constant $a(x)$.

#### 3. Stokes Flow with a Cylindrical Obstacle
Goal: Learn the mapping from in-flow velocity ${\bf u}_0 = (a(x), 0)$ to the pressure field $u(x)$.

#### 4. Inverse Discontinuity Coefficient in Darcy’s Flow

We also consider the inverse problem of reconstructing the **piecewise-constant** permeability field $a(x)$ from **sparse, noisy** observations of $u$. This problem has important applications in subsurface modeling and medical imaging.

## 🔗 Data Availability
- **All Physics-aware DNOs** in this repository are trained exclusively using physics information (i.e., **without labeled (a, u) pairs**).
- Training data (only for data-driven DNOs) and testing data can be downloaded from **[Google Drive](https://drive.google.com/drive/folders/1MOFme5DgUd339rlL1IGq35ZcVCR0CWqa?usp=drive_link)**.


## 📖 Citation
```
@article{zang2025dgenno,
  title={DGenNO: a novel physics-aware neural operator for solving forward and inverse PDE problems based on deep, generative probabilistic modeling},
  author={Zang, Yaohua and Koutsourelakis, Phaedon-Stelios},
  journal={Journal of Computational Physics},
  volume={538},
  pages={114137},
  year={2025},
  publisher={Elsevier}
}
```

