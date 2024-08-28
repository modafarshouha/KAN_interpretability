# KAN_interpretability
This is the code implementation for our paper: Is KAN More Interpretable Than MLP? A Comparative Study on Image and Text Data. <br>
The link to the paper will be added soon.

## Acknowledgement
In this work we compare MLPs to KANs. To do this, we use two KAN's variations; i.e. [Efficient-KAN](https://github.com/Blealtan/efficient-kan) and [ChebyKAN](https://github.com/SynodicMonth/ChebyKAN).

## Paper summary
We discussed the limitations of KAN's interpretability when dealing with high-dimensional data. We established that while KAN authors have not explicitly defined "interpretability", they generally considered smaller networks to be more interpretable. We arrived to this conclusion after studying their theoretical claims and the provided examples. Moreover, we highlighted the main difference between the underlying theorems of KANs (KAT) and MLPs (UAT). We showed that KAN was built on a generalized version of KAT theorem, which resulted in violating the network size constraint in it. Thus, KAN became neither far from MLP nor more interpretable. Furthermore, we highlighted that the original KAN implementation is not feasible for high-dimensional data; e.g. image and text. Hence, other variations of KAN were proposed; e.g. Efficient-KAN and ChebyKAN. We demonstrated through several experiments that these implementations were significantly larger than MLPs which made them less interpretable. This resulted in a considerable resource overhead without benefiting the accuracy; i.e. KAN's variations and MLPs performed similarly. Additionally, we highlighted that in some cases KAN had stability issues. For instance, we noticed that ChebyKAN convergence was not guaranteed and it was highly influenced by the network depth and the layers' size. In overall, we conclude that, next to other serious drawbacks, KAN fails to fulfill the interpretability promise; i.e. KAN is not more interpretable than MLP, mainly when dealing with high-dimensional data.

## Usage
The source code can be found (and imported from) [src](./src/) folder. <br>
All the expirements notebooks are available in [notebooks](./notebooks/) folder. <br>
For the data, in addition to the well-known MNIST data set, we used AG_NEWS data set. The latter went thorugh some preprocessing. We provide the preprocessing notebook in [data](./data/) folder. <br>

## Cite
```
will be provided soon...
```

## License
This work is licensed by <a href="./LICENSE">MIT License</a>.
