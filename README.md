# Project 11 - Coral Reef Detection



## Problem Statement

Coral Reefs are incredibly diverse ecosystems that support more species per unit area than any other marine environment. They also provide human protection, acting as a shoreline buffer from 97 percent of energy from waves, stores, and floods preserving life, property damage, and erosion [OA]. Currently, this ecosystem is facing a wide range of pressures, ranging from increased greenhouse gas emissions, human-driven fishing and shipping activities, and pollution and disease [RR]. While there are many organizations working to restore coral reefs, it is difficult to manually measure progress of coral growth.

## Dataset

The Coral Restoration Foundation (CRF) of Key Largo, FL provided us with 16 mosaic images, keys, and scale bars. The link to the dataset can be found here: https://drive.google.com/drive/u/2/folders/1LFDtqLMXoTlmAB7HkQt-IWURwc4ogRGH

To process the data in a usable format, we patchified each image into multiple 256x256 patches. At the end of this process, we have 197,248 patches. Then, we isolated images that contain 5% or more useful area. In this context, "useful area" is defined as a patch whose labels contain 5% or more values other than 0 (black space). After filtering, we have a total of 4,856 images. Split by classification, we have 2,661 Staghorn images and 2,195 Elkhorn images.

To further process the masks, we apply pixel segmentation such that Staghorn masks contain values [0, 1] and Elkhorn masks contain values [0, 2]. These are saved in a .png format.

## Developed Models

### U-Net Architecture For Coral Mask Prediction
The goal of this model is to generate coral masks that correspond with the location, shape, and size of corals in a still image. This will allow development of a pipeline that can automatically generate coral mosaic keys, rather than relying on hand-crafted keys. These keys can then be used to locate, count, and measure corals in large-scale mosaics. To generate these masks, we find semantic segmentation to best fit the needs of CRF. Semantic segmentation is a pixel-based classification method in which each pixel in an image is assigned a particular class. In our application, these classes include 0 for background, 1 for Staghorn, and 2 for Elkhorn.

To implement semantic segmentation, we utilized U-Net architecture. U-Net is a convolutional network architecture that is used for semantic segmentation and was first implemented in the Biomedical field for medical image screening purposes. U-Net consists of a downsampling path, a bottleneck, and an upsampling path. We use softmax activation, resnet34 as a backbone, and imagenet as encoder weights. We also use Adam optimizer in our implementation. 

For our final run included in this project, we use a learning rate of 0.0001, a batch size of 8, 25 epochs, and a verbose of 1. We applied data augmentation on both the training and testing set, including an increased contrast of 1.3, brightness of 1.1, and sharpness of 2. These images were normalized. Our measurement of accuracy for this model is IoU, or intersection of union. This lays the predicted mask over the ground truth mask and calculates the area of overlapping masks divided by the area of union.

Our final results are as follows:
- Mean IoU = 0.72698855
- IoU class 0 (background) = 0.88367414
- IoU class 1 (Staghorn) = 0.5304157
- IoU class 2  (Elkhorn) = 0.76687604

### ViT Classification
We use Vision Transformers to classify corals as Elkhorn or Staghorn. Transformers can find use in
Natural Language Processing. When using ViT, a given image is divided into patches and passed into
the model linearly, then transformed by being passed into a specified vector size. All the vectors are
passed to the main Transformer block where we normalize the layers, then we pass the vectors as key,
value, and query. For each element in an input sequence, we compute a weighted sum over all values
”v” in the sequence. The attention weights are based on the pair-wise similarity between two elements
of the sequence and their respective query and key representations. The Transformer Encoder step is
repeated 8 times, then we add a Dense layer.

Here, we use a training-testing-validation split of 60%, 20%, and 15%. We use a learning rate
of 0.001 and 150 epochs with the Adam optimizer. We apply data augmentation to some randomly
selected images, including rotating, flipping, zooming, and resizing. For accuracy metrics, we used
SparseCategoricalAccuracy and SparseTopKCategoricalAccuracy.

### YOLOV5 Object Detection
YOLOV5 is an object detection model. For a given image, it predicts whether a coral belongs to
class Elkhorn or Staghorn, then places bounding boxes around them. To prepare the labels, we used
the website Make Sense, which outputs the labeled data needed for YOLOV5. We upload images to
the website and manually add bounding boxers around corals and label them as either Elkhorn or
Staghorn. For this model, we ran with a batch size of 16 with 50 epochs. Our results are as follows:

- Precision: 0.773 Elkhorn, 0.825 Staghorn
- Recall: 0.805 Elkhorn, 0.778 Staghorn
- Mean Average Precision: 0.804 Elkhorn, 0.835 Staghorn

### Watershed Coral Counting
This section consists of information regarding experimentation and does not have any accuracy measurements associated with it. Here, we attempt to count corals using the Watershed algorithm. This
algorithm takes an input of gray scale images where the high intensity denotes peaks and the low
intensities denote valleys.
For each image, we apply sharpening and a Laplacian filter. As a binary image, we can apply
distance transform on the image using L2 norm and normalize them. We extract peaks using a
dilation operation and draw contours on the image. After running this algorithm across all mosaics,
we estimate that there are 915 Elkhorn corals and 1657 Staghorn corals in total.

## Sources
[OA] National Oceanic and Atmospheric Administration. The importance of coral reefs. National
Ocean Service.
[RR] Hannah Ritchie and Max Roser. Coral reefs. Our World In Data.
