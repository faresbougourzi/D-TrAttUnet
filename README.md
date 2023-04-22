# D-TrAttUnet: Dual-Decoder Transformer-Based Attention Unet  Architecture for Binary and Multi-classes Covid-19 Infection Segmentation.

In summary, the main contributions of this paper are as follows:

- A hybrid CNN-Transformer network is proposed that leverages the strengths of Transformers and CNNs to extract high-level features during the encoding phase.

- The proposed D-TrAttUnet encoder consists of two paths; the Transformer path and the Unet-like Fusion path. The Transformer path considers 2D patches of the input image as input, and consecutive Transformer layers to extract high representations at different levels. Four different Transformer features at different levels are injected into the Unet-like Fusion Encoder through UpResBlocks. On the other hand, the first layer of the Unet-like path uses the convolutional layers on the input image. The following Unet-Like Fusion layers combine the Transformer features with the previous layer of the Unet-Like path through concatenation and ResBlocks.

- The proposed D-TrAttUnet decoder consists of dual identical decoders. The objective of using two decoders is to segment Covid-19 infection and the lung regions simultaneously. Each decoder has four Attention Gates (AG), ResBlocks and bilinear Upsampling  layers similar to the Attion Unet (AttUnet) architecture, taking advantage of CNN-Transformer and multi-task tricks.


- To evaluate the performance of our proposed architecture, both binary infection segmentation and multi-classes infection segmentation are investigated using three publicly available datasets. 

- The comparison with three baseline architectures (Unet \cite{ronneberger_u-net_2015}, Att-Unet \cite{oktay_attention_2018}, and Unet++ \cite{zhou_unet_2018}) and three state-of-the-art architectures for Covid-19 segmentation (CopleNet \cite{wang_noise-robust_2020}, AnamNet \cite{paluru_anam-net_2021}, and SCOATNet \cite{zhao2021scoat}), demonstrates the superiority of our proposed D-TrAttUnet architecture in both Binary and Multi-classes Segmentation tasks.

![General drawio (1)](https://user-images.githubusercontent.com/18519110/233785211-8edaf09f-1fef-4b4e-9b76-9870b8bea9a9.png)
<img src="[https://i.imgur.com/ZWnhY9T.png](https://user-images.githubusercontent.com/18519110/233785211-8edaf09f-1fef-4b4e-9b76-9870b8bea9a9.png)" width="150" height="280">

<p align="center">
  Figure 1: The summary of our proposed D-TrAttUnet approach.
</p> 

![D-TrAttUnet drawio](https://user-images.githubusercontent.com/18519110/233785158-3c28b2c2-5872-42b4-bfb2-f37eb9a2183f.png)


<p align="center">
  Figure 2: Detailed Structure of the proposed D-TrAttUnet architecture.
</p> 

## Implementation:
#### D-TrAttUnet architecture:
``` Architectures.py ``` contains the implementation of the proposed D-TrAttUnet architecture.


#### Training and Testing Implementation:
``` detailed train and test ``` contains the training and testing implementation.

- First: the dataset should be prepared using ``` prepare_dataset.py ```, this saves the input slices, lung mask, and infection mask as ``` .pt ``` files
The datasets could be donwloaded from: http://medicalsegmentation.com/covid19/

- Second:  ``` train_test_PDAttUnet_BinarySegmentation.py ``` can be used to train and test the D-TrAttUnet architecture for the Binary Segmentation Task.


## Citation: If you found this Repository useful, please cite:

```bash
@article{bougourzi2023d,
  title={D-TrAttUnet: Dual-Decoder Transformer-Based Attention Unet Architecture for Binary and Multi-classes Covid-19 Infection Segmentation},
  author={Bougourzi, Fares and Distante, Cosimo and Dornaika, Fadi and Taleb-Ahmed, Abdelmalik},
  journal={arXiv preprint arXiv:2303.15576},
  year={2023}
}
```
 

