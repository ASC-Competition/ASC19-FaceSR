# ASC19-FaceSR
The validation data and testing data for ASC19 face super resolution challenge. https://www.asc-events.org/ASC19/


## Description about the challenge

**Goal**: Face Super Resolution (FSR), also known as face hallucination, is a domain-specific super-resolution problem. As a specific problem of Super-Resolution (SR), the aim of FSR is to generate high-resolution (HR) face images from low-resolution (LR) face images. One of the ultimate goals in FSR is to explore image intensity correspondences between LR and HR faces from large scale dataset and generate HR face images closed to the ground truth HR face images. In the final competition, the participant should design/tuning their algorithm designed in the preliminary competition to do the 4x FSR upscaling for face images which were down-sampled with a bicubic kernel. For instance, the resolution of a 400x600 image after 4x upscaling is 1600x2400. An example is given below, left is HR face image which resolution is 128x128, and right is the 4x down-sampling image which resolution is 32x32.


![image](https://github.com/ASC-SSC/ASC19-FaceSR/blob/master/img/1.png)


a)	On the spot in the final competition, the committee will supply scoring script, training dataset and test dataset. all test-dataset face images have identical resolution. 

b)	Each team should submit all of the reconstructed high-resolution face images of test dataset for scoring test. The goal is to achieve the identity similarity (IS) value close to 1. IS is the cosine similarity of the two feature vectors of the HR face and SR face, while the feature vector is extracted from the 512-D embedding feature of SphereFace model (https://github.com/clcarwin/sphereface_pytorch ).

c)	Each team is required to use PyTorch for this task. Any other deep learning framework will be prohibited. 



## Description about the data
There are 300 images for validation and 300 images for scoring. The high resoultion for images is 96x112, and 4x down-sampling low resolution images is 24x28.


## Scoring the super resolution results
```
1. Download the pre-trained sphereface network from https://github.com/clcarwin/sphereface_pytorch
2. python evaluate.py --model shpereface_model.pth --HR_dir <> --SR_dir <>
```

