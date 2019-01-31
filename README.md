# Skin Cancer Classification Using Deep Learning

Code for my replication of one of the best performing solutions to the ISIC challenge. Challenge Leaderboard: https://challenge2018.isic-archive.com/leaderboards/

## Summary

Recently, I worked on a project with [Dr. Qian](http://www.ece.tamu.edu/~xqian/) to classify images of skin lesions into the type of skin cancer they exhibited. The data was provided by the International Skin Imaging Collaboration (ISIC) from the [HAM10000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T). The [ISIC Challenge 2018](https://challenge2018.isic-archive.com/) consisted of 3 tasks. This post is aimed at tackling Task 3 - Disease Classification from images of skin lesions.

I assume if you're reading this, you know what a Convolutional Neural Network (CNN) is. There are some CNN architectures that have shown to historically perform well on image classification tasks. It would make sense to try those out first. I started by using a simple [VGG19](https://arxiv.org/abs/1409.1556) architecture pretrained on ImageNet and then tried the [ResNet50](https://arxiv.org/abs/1512.03385), replacing the last 2 layers with new fully connected layers. ResNet50 performed much better, so I used it with further tuning to achieve results on par with the best on the [leaderboard](https://challenge2018.isic-archive.com/leaderboards/).

You can find the code used for the post below in my 3 notebooks:

1. [Loading Data](Task3-Load_Imgs.ipynb)
2. [Training](Task3-Training.ipynb)
3. [Prediction](Task3-Prediction.ipynb)

## Introduction

In the United States, 5 million new cases of skin cancer are diagnosed every year. Of these, melanoma which is the deadliest accounts for over 9000. The diagnosis via visual inspection by patients and dermatologists is accurate only about 60% of the time. Moreover, the shortage of dermatologists per capita has abetted the need for computer-aided methods to detect skin cancer.

The HAM1000 dataset is an aggregation of a large amount of publicly accessible dermoscopy images labeled with ground truth data. The ISIC 2018 challenge was divided into 3 tasks -

1. Lesion Segmentation
2. Lesion Attribute Detection
3. Disease Classification. 

I worked on Task 3 i.e. classification of images into one of 7 possible classes and will detail the process in this post.

## Dataset

There are 10,015 images in the labeled training dataset. Some sample images from the dataset and their labels are shown below.

<div style={margin:auto;}>

![](/img/Task3-Imgs.png)

##### Lesion Images

</div>

The labels are stored in a CSV file in the form of stacked transposes of one-hot vectors. i.e. each example in the dataset is represented by a row of length 7 with only the class to which the exmple belogns being 1 and the other elements in the row being 0. There are no missing
labels and all images are classified into one of 7 classes: 

* Melanoma
* Melanocytic nevus 
* Basal cell carcinoma
* Actinic keratosis / Bowen's disease (intraepithelial carcinoma)
* Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis) 
* Dermatofibroma
* Vascular lesion

Evaluation metric for this task is the multi-class accuracy (MCA) i.e. the average of precision for all classes. 

<center>

![](/img/mca.png)

</center>

where $P_i$ is the precision of class $i$ and $n$ is the number of classes

## Architecture 
Since this is a problem of image classification, a convolutional neural network (CNN) architecture would be most suitable. I found a good implementation of a bunch of CNN architectures in Tensorflow, in this GitHub repo called [Tensornets](https://github.com/taehoonlee/tensornets)

I first used the simple VGG19 model with weights pretrained on the [ImageNet](http://www.image-net.org/) dataset. ImageNet is a huge dataset of labeled images classified into 1000 classes. Here I just needed 7 classes for the 7 types of skin cancer. So the last layer had to be replaced by a Softmax classifier with 7 outputs. I retrained this layer but it performed rather poorly on the MCA metric defined above. I then tried the much more sophisticated ResNet50 architecture, also pretrained on ImageNet. It performed much better on the MCA metric.

Once I had chosen to go ahead with ResNet50, I removed all the fully connected layers in the architecture and replaced them with 2 fully connected layers of 120 and 84 units respectively. Each new layer used a ReLU activation. Now I could train only these layers to perform the classification task.

Since the pretrained model I used had an input size of 224x224 and the images in the dataset are 450x600, the first step was to scale the images down to the required 224x224 size. Finally, a randomly selected sample of 10% of the dataset was used as a validation set to calculate the MCA and further tune the model.

## Improving Performance
As shown in the figure below, approximately 70% of the images belong to only one class (Melanocytic nevus). 

<center>

![](/img/Task3-DiseaseTypeFrequency.png)

##### Disease Type Frequency

</center>

Hence, it is trivial to achieve around 70% accuracy by simply predicting all images to be of that class. That is obviously incorrect. In order to improve performance, I tried several techniques such as data augmentation, oversampling low-frequency classes, weighted loss etc.

As expected a baseline ResNet50 model achieved a validation MCA of only 62.74% MCA while the training MCA was 93.67%. To improve this discrepancy in the performance of training and validation set I used some techniques as follows.

### Batch Normalization
Since we are using ReLU activation in the fully connected layers and the final output is a softmax i.e. a number between 0 and 1, batch
normalization could help speed up training by scaling the output of the fully connected layers appropriately. The performance on the validation set improved slightly to 73.29% but the training MCA went down to 87.77%. This is expected as Batch Normalization has a slight regularization effect.

### Data Augmentation - Mirroring
There is still a large difference between the validation and training MCA. Training the model on a larger dataset could help bridge this gap. We can double the dataset by simply taking mirror images of the existing dataset while keeping the labels constant. Training the model on the dataset with original imges and their horizontal mirror images increased the validation MCA to 76.27% while the training MCA was up to 92.85%

<center>

![](/img/Task3-Imgs-Mirror.png)

##### Horizontally Flipped Images

</center>

### Weighted Loss
The model still predicts the dominating class more often than it should while ignoring lesser occuring classes. One way to fix this is to penalize the model for predicting the dominating class. This can be done by multiplying the loss function by the frequency of classes. Thus, a new weighted loss function can be used to train the model.

The weights for the loss function are calculated as shown below.

<div>$$w_i = \frac{1}{m}\sum_{j=1}^{m}Y_{ij}$$</div>

where $Y\_{ij}$ is the value of class $i$ in example $j$ and $m$ is the number of examples in the original training set. Since $Y\_j$ is a one-hot vector, the value of $Y\_{ij}$ is either 0 or 1. The mean of this along the number of examples gives the frequency of class $i$ in the dataset. The calculated weights are shown in the table below


<center>

##### Table: Weights for Loss Function By Class

$i$ | Class | Weight ($w_i$)
:---|:---|---:
0 |Melanoma | 0.111
1 |Melanocytic nevus | 0.669
2 |Basal cell carcinoma | 0.051
3 |Actinic keratosis | 0.033
4 |Benign keratosis | 0.109
5 |Dermatofibroma | 0.011
6 |Vascular lesion | 0.014

</center>

Now, the new value of loss function can be written as shown in the equation below.


<div>
$$J = (\sum_{j=1}^{b}w_iY_{ij}).(\frac{1}{b}\sum_{j=1}^{b}\sum_{i=0}^{n-1}Y_{ij}\log\hat{Y}_{ij})$$
</div>

where $w\_i$ is the weight as calculated above, $b$ is the size of the batch used to run backpropagation (using the Adam optimizer) and $\hat{Y}\_{ij}$ is the softmax probability of class $i$ predicted for example $j$ by the model. As a result of this multiplication, the classes occuring more frequently are penalized with a higher loss function whereas those that occur less frequently are rewarded with a lower loss function. Training the model with the weighted loss function got a validation MCA of 75.73% and training MCA of 95.25%.

### Oversampling

The presence of classes Dermatofibroma and Vascular lesion (class 5 and 6) is very low in the dataset (approx 1%). We can increase their occurence by taking random crops of the central part of the image so that the lesion still remains in the image. I took 4 random crops of images belonging to these classes and also their horizontal mirror images while keeping the labels constant. Also I took vertical mirror images of images belonging to the Actinic keratosis (class 3) which also occurs less frequently. All these new images were then added to the dataset alongwith the original labels. From this, 90% of the data was randomly selected for training. The validation MCA shot up to 87.47% as a result while training MCA was 98.08%.

<center>

![](/img/Task3-Imgs-DF-Crops.png)

##### Random Crops

</center>

## Results

The results achieved from training using the techniques listed above are
shown in the table below.

<center>

##### Table: Effect On Model Performance

Technique | Training MCA | Validation MCA
:---|:---:|:---:
Baseline | 93.67% | 62.74%
Batch Normalization | 87.77% | 73.29%
Data Augmentation - Mirroring | 92.85% | 76.27%
Weighted Loss | 95.25% | 75.73%
Oversampling | 98.08% | 87.47%

</center>

## Conclusion and Discussion

The ResNet50 architecture with data augmentation is able to perform much better than the baseline but there is still a difference between the training and validation metrics. The best training MCA is over 98% but the best validation MCA is about 87.5%. Since the training and  validation sets are randomly selected from the same dataset, the chances of data mismatch are minimal. The training-validation gap may be converged further by training on a larger dataset. Also, there is a possibility of illumination affecting the database which can be corrected using color constancy on the entire dataset.

## Acknowledgements

I would like to thank the Texas A&M High Performance Research Computing (HPRC) for providing the necessary computational resources. Also, I would like to thank Taehoon Lee who trained several models on the ImageNet dataset and provided an open source implementation in Tensorflow.