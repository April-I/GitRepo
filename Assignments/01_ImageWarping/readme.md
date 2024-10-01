# Assignment 1 - Image Warping

## 问题设置

### 1. 基本图像操作：缩放、旋转和平移。
补全'run_global_transform.py'的[缺失部分](run_global_transform.py#L21) 

### 2. 基于点的图像变形。
在“run_point_transform.py”的[缺失部分](run_point_transform.py#L52)中实现基于MLS或RBF的图像变形。

## 算法设计

### 1. 基本图像操作
由于围绕图像中心进行放缩和旋转，先确定中心坐标
$$
x_{center}=\frac{image.shape[1]}{2}\\
y_{center}=\frac{image.shape[0]}{2}
$$

不妨现将中心移到(0,0)进行后续操作
$$
\begin{bmatrix}
x_{new} \\
y_{new} 
\end{bmatrix}
=\begin{bmatrix}
    1 & 0 & -x_{center}\\
    0 & 1 & -y_{center}
\end{bmatrix}
\begin{bmatrix}
    x\\
    y\\
    1
\end{bmatrix}
$$

因此缩放操作为
$$
x_{new}=scale*x\\
y_{new}=scale*y
$$

写成矩阵形式为
$$
\begin{bmatrix}
x_{new} \\
y_{new} 
\end{bmatrix}
=\begin{bmatrix}
    scale & 0\\
    0 & scale
\end{bmatrix}
\begin{bmatrix}
    x\\
    y
\end{bmatrix}
$$

同理旋转操作为
$$
x_{new}=\cos{\theta}*x-\sin{\theta}*y\\
y_{new}=\sin{\theta}*x+\cos{\theta}*y
$$

写成矩阵形式为
$$
\begin{bmatrix}
x_{new} \\
y_{new} 
\end{bmatrix}
=\begin{bmatrix}
    \cos{\theta} & -\sin{\theta}\\
    \sin{\theta} & \cos{\theta}
\end{bmatrix}
\begin{bmatrix}
    x\\
    y
\end{bmatrix}
$$

对于平移操作translation_x, translation_y，即
$$
x_{new}=x+translation_x\\
y_{new}=y+translation_y
$$

写成矩阵形式为
$$
\begin{bmatrix}
x_{new} \\
y_{new} 
\end{bmatrix}
=\begin{bmatrix}
    1 & 0 & translation_x\\
    0 & 1 & translation_y
\end{bmatrix}
\begin{bmatrix}
    x\\
    y\\
    1
\end{bmatrix}
$$

最后再将图像移回原中心
$$
\begin{bmatrix}
x_{new} \\
y_{new} 
\end{bmatrix}
=\begin{bmatrix}
    1 & 0 & x_{center}\\
    0 & 1 & y_{center}
\end{bmatrix}
\begin{bmatrix}
    x\\
    y\\
    1
\end{bmatrix}
$$

### 2. 基于点的图像变形：MLS方法
图像变形问题可转化为对于图像中的任一像素点v，求解一个映射函数$l_v(x)$使下式最小化能量函数
$$
E=\sum\limits_iw_i|l_v(p_i)-q_i|^2
$$
其中p是控制点的集合，q是控制点p的变形位置，权重$w_i$定义为
$$
w_i=\frac{1}{|p_i-v|^{2\alpha}}
$$
由于$l_v(x)$是一个仿射变换，所以$l_v(x)$由两个部分组成：一个线性变换矩阵$M$和一个平移矩阵$T$
$$
l_v(x)=xM+T
$$
代入能量函数可得
$$
E=\sum\limits_iw_i|p_iM+T-q_i|^2
$$
对T求导并取0
$$
\frac{\partial E}{\partial T}=2\sum\limits_iw_i(p_iM+T-q_i)=0
$$
可以得到T为
$$
T=q_{*}-p_{*}M
$$
其中
$$
p_{*}=\frac{\sum\limits_iw_ip_i}{\sum\limits_iw_i},\ q_{*}=\frac{\sum\limits_iw_iq_i}{\sum\limits_iw_i}
$$

代回能量可得
$$
E=\sum\limits_iw_i|p_iM+q_{*}-p_{*}M-q_i|^2=\sum\limits_iw_i|(p_i-p_{*})M-(q_i-q_{*})|^2
$$
令$\hat{p_i}=p_i-p_{*},\hat{q_i}=q_i-q_{*}$，则能量可写为
$$
E=\sum\limits_iw_i|\hat{p_i}M-\hat{q_i}|^2
$$
再对M求导，并令导数等于0，可得到
$$
M=\frac{\sum\limits_j\hat{p}_j^Tw_j\hat{q}_j}{\sum\limits_i\hat{p}_i^Tw_i\hat{p}_i}
$$

由此可以得到仿射函数
$$
l_v(v)=xM+T=(v-p_*)\frac{\sum\limits_j\hat{p}_j^Tw_j\hat{q}_j}{\sum\limits_i\hat{p}_i^Tw_i\hat{p}_i}+q_*
$$

又由$p_i$是固定值，因此可以预先计算出矩阵A_j
$$
A_j=(v-p_*)\frac{\hat{p}_j^Tw_j}{\sum\limits_i\hat{p}_i^Tw_i\hat{p}_i}
$$
所以函数可写为
$$
l_v(v)=\sum\limits_jA_j\hat{q}_j+q_*
$$

若满足相似变换的性质，则需限制$M$满足$\exists\lambda\ \ s.t.\ \ M^TM=\lambda^2I$，并令分块矩阵M写为
$$
M=(M_1\ \ M_2)
$$
其中$M_1,M_2$是长度为2的列向量，因此可以得到$M_1^TM_1=M_2^TM_2=\lambda^2$且$M_1^TM_2=0$，即$M_2=M_1^{\bot}$，对于二维向量$(x,y)^{\bot}=(-y,x)$，因此原来的能量方程可以写为
$$
\sum\limits_iw_i|
    \begin{pmatrix}
    \hat{p}_i\\
    -\hat{p}_i^{\bot}
    \end{pmatrix}
    M_1-\hat{q}_i^T|^2
$$
由此得到最优M
$$
M=\frac{1}{\mu_s}\sum\limits_iw_i
    \begin{pmatrix}
    \hat{p}_i\\
    -\hat{p}_i^{\bot}
    \end{pmatrix}
    (\hat{q}_i^T\ \ \hat{q}_i^{\bot T})
$$
其中$\mu_s=\sum\limits_iw_i\hat{p}_i\hat{p}_i^T$。由此得到变换公式为
$$
f_s(v)=\sum\limits_i\hat{q}_i(\frac{1}{\mu_s}A_i)+q_*
$$
其中
$$
A_i=w_i\begin{pmatrix}
    \hat{p}_i\\
    -\hat{p}_i^{\bot}
\end{pmatrix}
\begin{pmatrix}
    v-p_*\\
    -(v-p_*)^{\bot}
\end{pmatrix}^T
$$
进一步限制$M^TM=I$，即只需将上式的$\mu_s$替换为$\mu_r$
$$
\mu_r=\sqrt{(\sum\limits_iw_i\hat{q}_i\hat{p}_i^T)^2+(\sum\limits_iw_i\hat{q}_i\hat{p}_i^{\bot T})^2}
$$
令
$$
\vec{f_r}(v)=\sum\limits_i\hat{q}_iA_i
$$
其中$A_i$仍由上述定义，则得到最终变换公式为
$$
f_r(v)=|v-p_*|\frac{\vec{f}_r(v)}{|\vec{f}_r(v)|}+q_*.
$$

## 实验结果
### Requirements
To install requirements:

```setup
python -m pip install -r requirements.txt
```

### Running

To run basic transformation, run:

```basic
python run_global_transform.py
```

To run point guided transformation, run:

```point
python run_point_transform.py
```

### 1. 基本图像操作
<img src="pics/global_demo.gif" alt="alt text" width="800">

### 2. 图像变形操作
<img src="pics/1.png" alt="alt text" width="400"><img src="pics/2.png" alt="alt text" width="400">

<img src="pics/3.png" alt="alt text" width="400"><img src="pics/4.png" alt="alt text" width="400">

<img src="pics/5.png" alt="alt text" width="400"><img src="pics/6.png" alt="alt text" width="400">