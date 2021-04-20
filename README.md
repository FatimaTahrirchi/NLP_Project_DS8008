# Text Preprocessing on VDCNN Architecture

## Introduction:

#### Problem Description:

NLP tasks are typically accomplished using recurrent neural networks and convolutional networks that tend to be shallow implementations with processing granularity that can range from individual characters, to words or even whole sentences, in order to most effectively analyze and extract relevant information.  The key objective of this study is to evaluate network architectures with very large depths and using small temporal convolution filters with different types of pooling against shallow architectures.  The expectation is that very deep convolutional network architectures may yield better performance than their shallow counterparts.

#### Context of the Problem:

The key determining factor for employing the use of very deep concolutional neural networks (VDCNN) was the success that this type of architecture has in the field of computer vision because of the compositional structure of an image.  It has been posited that texts have similar heirarchical properties (given that characters combine to form n-grams, words, sentences and so on) and therefore the solution may be tranferable across domains (Conneau et. al., 2017).  Therefore, developing a neural network with a deep architecture should yield more favorable results over traditional, shallow architecture implementations.
The objective of the research paper that serves as a basis for this study (Conneau et. al., 2017) is to evaluate network architectures with large depths and using small temporal convolution filters with different types of pooling against shallow architectures with the expectation that very deep convolutional network architectures may yield better performance than their shallow counterparts.

#### Solution:
In order to implement a novel solution architecture to a familiar problem is to look to not only past innovations but also to other domains where that may have transferable similarities to this problem space.  Conneau, et. al. (2017) have presented an interesting and effective solution to the text processing space based on an architecture that is typically used in computer vision applications.  However, while the paper discusses the implementation at length, decided to study this implementation and determine whether combining specific data preprocessing steps will further contribute to effectiveness of this architecture.

# Background

Explain the related work using the following table

| Reference |Explanation |Dataset/Input |Weakness
| --- | --- | --- | --- |
|Sennrich et al., 2016. [1] |This paper discussed a new approach to the translation of out-of-vocabulary words with the use of a simpler and more effective approach by encoding rare and unknown words as sequences of sub-word units.  This approach is based on the idea that word classes are translatable through smaller units than words.  The suitability of different word techniques are discussed (e.g. n-gram models) to show that subword models are an improvement over traditional (back-off dictionary) approaches.| WMT2015 Dataset (English->German, English->Russian) |Poor translation results if vocabulary size of subword models is too large
|Moriya, Sh et al. 2018. [2] |The effective of  using transfer learning between datasets that have a certain degree of similarity. The key motivation for this approach was given the challenge of finding the volume of labeled data necessary in order to train CNNs.  Given that this approach has been successful for image recognition tasks, expectation is for promising results in the text processing domain.  Experimental results revealed that the full transfer method (vs. non-transer and partial transfer) is the most effective method.|AFPBB|Weakness in this study is the limitation of datasets that were used for testing purposes.
|Conneau, A et al. 2017. [3] | Implementation of a very deep convolutional neural network, which is the first implementation of a deep architecture of this kind.  The motivation for this approach was primarily from the image processing domain.  Other architectural features were adopted from related research studies that yielded positive results (e.g. k-max pooling, character-level embedding).| AG news, Sogou news, DBPedia, Yelp Reviews, Amazon review, Yahoo! Answers | The research study did not use any preprocessing (except lowercasing) 

# Methodology

The proposed architecture is developed based on studies of related works, which includes the use of fixed size representations of words or characters into a low-dimensional space; the use of recursive and convolutional neural networks; and, the use of multiple temporal k-max pooling layers.
The approach is to generate a 2D tensor that contains embeddings of the characters fixed to 1024.  Subsequently, a layer of 64 convolutions of size 3 is applied, which are then followed by temporal convolutional blocks.  There are 3 pooling operations, which half the temporal resolution that results in feature maps of 128, 258 and 512 feature maps.  
For the classification, the temporal resolution of the output of the convolution blocks is first down-sampled to a fixed dimension using k-max pooling in order to extract the k most important features independent of where they appear in the sentence.  The resulting features are transformed into a single vector, which is input to a three-layer, fully connected classifier with ReLU hidden units and softmax outputs.  For all of the experiments, k is set to 8 and the number of hidden units to 2048.
Each convolution block is a sequence of two convolution layers with each one followed by a temporal BatchNorm layer and a ReLU activation.  The kernel size of all temporal convolutions is 3.  Between blocks, three types of down-sampling are tested, all of which reduce the temporal resolution by a factor of 2.  Refer to Figures 1 and 2 for a pictoral description of the architecture.

![Conv Block](https://github.com/FatimaTahrirchi/NLP_Project_DS8008/blob/main/Images/ConvBlock.png)
![Model](https://github.com/FatimaTahrirchi/NLP_Project_DS8008/blob/main/Images/VDCNN.png)
This paper explored four convolutional depths:  9, 17, 29, 49 (determined by summing the number of blocks with 64, 128, 256 and 512 filters, with each block containing two convolutional layers).  The observed outcomes is that the very deep network architectures performs better than the more shallower architectures.  In addition, those datasets with much more training samples (i.e. 3 million) markedly increased performance.  When the depth was increased from 9 to 49 layers, performance began dropping.  Although when using ‘shortcut’ connections in order to solve for the vanishing gradient issue, slightly improved results were observed across all depths.

A number of datasets were tested that ranged from 120K to 3.6M instances and with classes between 2 to 14.  Notable conclusions were as follows:
(1)	Depth improves performance – as the network depth increased, the test errors decreased on all data sets (e.g. going from depth 9 to 29 reduced the error rate by 1% when using the Amazon dataset – 3 million samples)
(2)	Going too deep degrades performance – gain in performance due to depth is limited; when the depth increases too much the accuracy of the model gets ‘saturated’ and starts to degrade rapidly, primarily due the challenge of optimizing a very deep model and also due to the gradient vanishing problem.  However, this limitation can be overcome by ‘shortcut’ connections between convolutional blocks that allow the gradients to flow more easily in the network.

Our proposal is to replicate this implementation and to determine if whether a focus on data preprocessing will provide additional incremental improvements to overall accuracy.  Our approach is to use the Yelp data (same as one of the datasets used in the original research study) and to evaluate data to determine it's effectiveness on sentiment analysis (i.e. polarity) and classification (5 classes in total).  While the original study used roughly 600k data samples, given resource limitations this study using roughly 60k but baselining the original implementation and basing incremental improvements on that baseline.  (Note that the original paper reports results of 94% for polarity and 62.7% for the classification exercise on this dataset). Preprocessing is mainly focused on the traditional data text normalizing steps (i.e. lowercasing, stemming, lemmatizing, etc.) but also incorporating other aspects such as focusing on specific parts-of-speech tags while filtering out others or adding additional features using positive and negative word banks.  The results of additional preprocessing steps resulted in the incremental benefits, as below:

![Prepration](https://github.com/FatimaTahrirchi/NLP_Project_DS8008/blob/main/Images/Table1.png)
![Result ](https://github.com/FatimaTahrirchi/NLP_Project_DS8008/blob/main/Images/Table2.png)

As indicated, given resource limitations, the first result in Table 1 serves as our baseline to determine whether any incremental improvements have been made.  Additional two use cases focuses on specific preprocessing techniques, which resulted in roughly 4-5% accuracy gains for both the sentiment / polarity exercise and for the 5-class classification problem, which is quite significant in the text processing domain.

# Conclusion and Future Direction

The results of additional preprocessing were promising as the Table 1 illustrates and even at that with using traditional approaches with low complexity.  A notable outcome of this study is that while the solution architecture has been the central focus of innovating improvements in the text processing space.  Additional focus on determining whether additional, possibly more complex, preprocessing techniques may be well-placed when paired with effective solution architectures in order to further enhance performance by yielding additional, incremental benefit in performance. 

