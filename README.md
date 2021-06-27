# SmartFlow

A Deep Learning solver for the Shallow Water Equations.
___

SmartFlow is an extend of [MattFlow], which is a 2-stage [Runge-Kutta] numerical
solver for the [Shallow Water Equations] (SWE). SmartFlow implements some Deep
Learning architectures out of the corresponding papers. The models are trained
on data produced by MattFlow, aiming is to predict the successive states of the
fluid.

| requirements         |
| -------------------- |
| python3              |
| tensorflow >= 2.2.0  |
| numpy >= 1.19.2      |
| matplotlib >= 3.3.2  |
| mattflow >= 1.3.4    |
| pandas >= 1.1.3      |

## Install

```shell
$ pip install smartflow
```

```shell
$ git clone https://github.com/ThanasisMattas/smartflow.git
```

## Dataset format

Each data example is the state of the fluid at some time-step and the
corresponding label is the state at the next time-step. A state comprises a
3D matrix, U, where the *channel* axis consists of the 3 state variables of the
SWE, populating the discretized 2D domain. The main idea is that the state
matrix can be regarded as a 3-channel image, where each pixel corresponds to
a cell of the mesh and each channel to a state variable.

## Architectures

* [Inception-v3]
* [Inception-ResNet-v2]
* [Inception-ResNet-like]
* [ResNet]
* [Simple CNN]
* [Fully-Connected NN]

## Reference papers

* He, K., Zhang, X., Ren, S., Sun, J. *Deep Residual Learning for Image Recognition*. 2015. arXiv: [1512.03385].
* Szegedy, C., Ioffe, S., Vanhoucke, V., Alemi, A. *Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning*. 2016. arXiv: [1512.00567].
* Ioffe, S., Szegedy, C. *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*. 2015. arXiv: [1502.03167].
* Szegedy, S., Vanhoucke, V., Ioffe, S., Shlens, J., Wojna, Z. *Rethinking the Inception Architecture for Computer Vision*. 2015. arXiv:[1512.00567].

## License

[GNU General Public License v3.0]

<br />

>(C) 2021, Athanasios Mattas<br />
>thanasismatt@gmail.com


[//]: # "links"

[MattFlow]: <https://github.com/ThanasisMattas/mattflow>
[Runge-Kutta]: <https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods>
[Shallow Water Equations]: <https://en.wikipedia.org/wiki/Shallow_water_equations>
[Inception-ResNet-v2]: <smartflow/archs/inception_resnet_v2.py>
[Inception-v3]: <smartflow/archs/inception_v3.py>
[Inception-ResNet-like]: <smartflow/archs/inception_resnet.py>
[Resnet]: <smartflow/archs/resnet.py>
[Simple CNN]: <smartflow/archs/cnn.py>
[Fully-Connected NN]: <smartflow/archs/fcnn.py>
[GNU General Public License v3.0]: <https://github.com/ThanasisMattas/mattflow/blob/master/COPYING>


[1512.03385]: <https://arxiv.org/abs/1512.03385>
[1512.00567]: <https://arxiv.org/abs/1512.00567>
[1502.03167]: <https://arxiv.org/abs/1502.03167>
[1512.00567]: <https://arxiv.org/abs/1512.00567>
