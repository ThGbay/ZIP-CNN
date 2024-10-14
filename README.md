# ZIP-CNN
[ZIP-CNN: Design Space Exploration for CNN Implementation within an MCU](https://dl.acm.org/doi/abs/10.1145/3691343)

This is material and results for the paper "ZIP-CNN: Design Space Exploration for CNN Implementation within an MCU". 
The repository contains figures and the Python code used to train the CNN. 
The files .h5 contains the weights of the corresponding CNN. 
Several measurements were conducted on 3 microcontrollers units (STM32L496ZG - STM32F446ZE - STM32F746G) during an inference of CNN. Results are available in the measurements folder.

Feel free to send an email if there are questions.

If you use ZIP-CNN data in a scientific publication, we would appreciate using the following citations:
```
@article{10.1145/3691343,
author = {Garbay, Thomas and Hachicha, Khalil and Dobias, Petr and Pinna, Andrea and Hocine, Karim and Dron, Wilfried and Lusich, Pedro and Khalis, Imane and Granado, Bertrand},
title = {ZIP-CNN: Design Space Exploration for CNN Implementation within a MCU},
year = {2024},
issue_date = {January 2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {24},
number = {1},
issn = {1539-9087},
url = {https://doi.org/10.1145/3691343},
doi = {10.1145/3691343},
abstract = {Embedded systems based on Microcontroller Units (MCUs) often gather significant quantities of data and solve various issues. Convolutional Neural Networks (CNNs) have proven their effectiveness in solving computer vision and natural language processing tasks. However, implementing CNNs within MCUs is challenging due to their high inference costs, which varies widely depending on hardware targets and CNN topologies. Despite state-of-the-art advancements, no efficient design space exploration solutions handle the wide variety of implementation solutions. In this article, we introduce the ZIP-CNN design space exploration methodology, which facilitates CNN implementation within MCUs. We developed a model that quantitatively estimates the latency, energy consumption, and memory space required to run a CNN within an MCU. This model accounts for algorithmic reductions such as knowledge distillation, pruning, or quantization and applies to any CNN topology. To demonstrate the efficiency of our methodology, we investigated LeNet5, ResNet8, and ResNet26 within three different MCUs. We made materials and supplementary results available in a GitHub repository: . The proposed method was empirically verified on three hardware targets running at 14 different operating frequencies. The three CNN topologies investigated were implemented in their default configuration in FP32, and also reduced with INT8 quantization, pruning at five different rates and with knowledge distillation. The estimates of our model are very reliable with an error of 3.29\% to 15.23\% for latency, 3.12\% to 10.34\% for energy consumption, and 1.95\% to 6.31\% for memory space. These results are based on on-device measurements.},
journal = {ACM Trans. Embed. Comput. Syst.},
month = sep,
articleno = {5},
numpages = {26},
keywords = {Design space exploration, neural networks, microcontroller units, tiny machine learning}
}
```
