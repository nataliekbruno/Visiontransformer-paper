# How Do Vision Transformers Work? 
Namuk Park, Songkuk Kim

## Introduction

Multi-head self-attentions (MSAs) are a core component of computer vision, however little is known the nature of MSAs. Traditionally, Convolutional Neural Networks (CNNs) have dominated this field, but Vision Transformers with MSAs are outperforming them. This paper seeks to better understand MSAs and how they contribute to the success of Vision Transformers (ViTs) by exploring key architectural components and behavioral properties in comparision to CNNs. 

**Problem and Motivation**

- Understand the properties of MSAs that allow them to perform well
- Determine the accuracy of the widely held belief that the success of MSAs is due to their weak inductive bias and capture of long range dependencies.

**Research Questions**

- What properties of MSAs do we need to improve optimization?
- Do MSAs act like CNNs?
- How can we harmonize MSAs with CNNs?
  
**Approach Overview**
  
To explore the properties of MSAs and how they behave different from CNNs, the authors consider:
- Loss landscape visualizations of ViTs and CNNs
- MSA Frequency Behavior
- Network Architecture
  
**Application**
  
- Hybrid model AlterNet 
  

## Approach

1. **Lost Landscape Analysis**

<img width="683" alt="Screenshot 2024-11-06 at 11 06 52 PM" src="https://github.com/user-attachments/assets/72ae93c6-ee29-490e-bdf9-2e9cc35c8072">

This analysis demonstrates that MSAs consistently flatten the loss landscape during training. ViTs create a smoother, more trainable neural networks compared to ResNet (CNNs), demonstated by their more direct paths and smaller eigenvalues. This is significant because while large eigenvalues impede NN training, MSAs can help NNs learn better representations by suppressing these larger values. 


2. **MSA Behavior Analysis**

<img width="708" alt="Screenshot 2024-11-06 at 11 21 30 PM" src="https://github.com/user-attachments/assets/6469d794-51f8-4dfe-8127-294788514e2e">

(a) ResNet (CNNs) amplifies high frequencies, while ViT reduces high frequencies. 

(b) ResNet struggles with high-frequency noise while ViT handles high frequency noise better. 

3. **Network Stage Analysis**

<img width="618" alt="Screenshot 2024-11-06 at 11 11 03 PM" src="https://github.com/user-attachments/assets/7362020d-47f8-4a54-9abf-5b8793fbcdb9">

This feature map variance analysis reveals: 

Multi-head Self-Attention (MSA) mechanisms consistently reduce the variance of feature map points, effectively acting as a stabilizing force in the network. In contrast, Convolutional Neural Networks (CNNs) tend to increase variance, creating more diverse but potentially less stable representations. This variance accumulation occurs progressively through neural network layers, with a notable pattern emerging: the feature map variance reaches peak levels at the end of each stage in ResNet architectures. This pattern suggests that strategic placement of MSA blocks could help manage and utilize this variance effectively.

<img width="427" alt="Screenshot 2024-11-06 at 11 12 10 PM" src="https://github.com/user-attachments/assets/cd532337-9d4e-4f63-9a01-46163a123cfb">

Multi-Stage behavior

Neural networks with multiple stages behave essentially like a series of smaller, connected models. Each stage develops its own specialized processing characteristics, rather than functioning as a uniform processing pipeline. This finding has significant implications for architecture design, as the performance impact of different components varies dramatically depending on their position within these stages. The research shows that this staged behavior is consistent across different architectures but manifests differently between CNNs and Vision Transformers.
  
## Application: AlterNet Architecture 


<img width="523" alt="Screenshot 2024-11-06 at 11 15 44 PM" src="https://github.com/user-attachments/assets/c5f389df-c709-4b06-9399-31464ef2811d">

Comparison of three different repeating patterns 

Based on the paper's insights about MSAs and CNNs exhibiting opposite behaviors, the researchers propose AlterNet, a model that leverages the complementary nature of these components. Since MSAs and Convs are low-pass and high-pass filters respectively, AlterNet strategically combines them by placing MSA blocks at the end of CNN stages. This design pattern naturally derives from the understanding that multi-stage neural networks behave like a series connection of small individual models, where components at stage endings play crucial roles in prediction.


<img width="624" alt="Screenshot 2024-11-06 at 11 16 24 PM" src="https://github.com/user-attachments/assets/ebb04f10-4958-44b5-b856-5273283cf2d5">

Detailed AlterNet architecture:
1. Progressive Scaling: MSAs in stages 1 to 4 systematically increase in complexity with 3, 6, 12, and 24 heads respectively
2. Strategic Placement: All stages except stage 1 end with MSA blocks
3. Balanced Processing: Alternates between traditional Conv blocks and attention mechanisms
4. Systematic Organization: Based on pre-activation ResNet-50 structure with strategic modifications

<img width="696" alt="Screenshot 2024-11-06 at 11 18 23 PM" src="https://github.com/user-attachments/assets/6ae43afc-d5f2-46e4-8e86-c7ca53216b0e">

Performance of AlterNet vs. CNNs and ViTs

## Key Findings

1. MSA's Impact on Accuracy and Generalization
   
The study reveals that Multi-head Self-Attention mechanisms improve neural network performance through two primary mechanisms. First, they consistently flatten loss landscapes, making the optimization process more stable and efficient. Second, they reduce the magnitude of Hessian eigenvalues, which directly contributes to better generalization capabilities. Importantly, these improvements stem primarily from the MSA's data-specific processing nature, rather than their ability to capture long-range dependencies as previously thought. This finding challenges the conventional wisdom about why Vision Transformers work effectively.

2. Complementary Behavior of MSAs and CNNs
   
One of the most significant discoveries is the fundamentally different yet complementary nature of MSAs and CNNs. MSAs function as low-pass filters, effectively smoothing and aggregating information across feature maps. In contrast, CNNs act as high-pass filters, emphasizing fine details and local patterns. This difference explains their varying robustness to noise: MSAs handle high-frequency noise effectively because they naturally filter it out, while CNNs struggle with such noise because they amplify it. This complementary relationship suggests that combining both approaches could lead to more robust architectures.

3. Multi-stage Network Functionality
   
The research demonstrates that multi-stage neural networks operate as a series of interconnected smaller models, each with distinct characteristics and roles. MSAs positioned at the end of stages play particularly crucial roles in overall network performance. Each stage contributes uniquely to the network's processing pipeline, with early stages focusing on basic feature extraction and later stages handling more complex feature integration. This understanding led to the development of more effective architectural patterns for combining MSAs and CNNs.


## Critical Analysis

1. Data Scale Dependecies
A crucial limitation emerges in the context of data scale. Vision Transformers encounter challenges with non-convex losses when working with small datasets. This problem naturally diminishes with larger datasets, but it represents a significant constraint for applications with limited data availability. The researchers found that loss landscape smoothing methods can help mitigate this issue, but it remains an important consideration for practical applications.

## Impacts 

**Theoretical Impacts**

This research fundamentally advances our understanding of how Vision Transformers work. By challenging the conventional wisdom about MSAs and demonstrating their complementary relationship with CNNs, the study opens new avenues for architectural innovation. The insights about loss landscape characteristics and the role of data specificity provide a stronger theoretical foundation for future development in computer vision architectures.

**Practical Impacts**

The findings translate directly into practical improvements for computer vision systems. The developed AlterNet architecture demonstrates superior performance in both small and large data regimes, offering a novel solution to implementing both CNNs and ViTs. The insights about stage-wise processing and the importance of MSA positioning provide clear guidelines for practitioners designing new architectures for specific applications.

**Future Directions**

The complementary nature of MSAs and CNNs suggests potential for even more sophisticated hybrid architectures. The understanding of stage-wise processing behaviors could lead to more efficient network designs. Additionally, the insights about loss landscape characteristics could inform the development of better training methods for deep learning models.

## Citation

Park, N., & Kim, S. (2022). How Do Vision Transformers Work? Published as a conference paper at ICLR 2022. arXiv:2202.06709.

## Resource Links

1. Paper Implementation: https://github.com/xxxnell/how-do-vits-work
2. Vision Transformer Introduction: https://arxiv.org/abs/2010.11929
3. Original Transformer Paper: https://arxiv.org/abs/1706.03762







