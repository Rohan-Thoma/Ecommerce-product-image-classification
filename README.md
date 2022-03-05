# Ecommerce-product-image-classification for CDiscount.com
Classifying various product images in to respective classes with multi-class classification of the images via deep learning.

## Detailed Blog explaining the whole case study is here:
* https://medium.com/@rohanvailalathoma/ecommerce-product-image-classification-for-cdiscount-com-ff2802d4636d

## Introduction
Cdiscount.com is the largest non-food e-commerce company in France. The company has a wide variety of products which range from TVs to trampolines, the list of products is still rapidly growing. In the year 2017 when this competition was held, Cdiscount.com had over 30 million products up for sale. This is a rapid growth from 10 million products before 2 years. Ensuring that so many products are well classified is a challenging task. At that time, Cdiscount.com applied machine learning algorithms to the text description of the products in order to automatically predict their category. As those methods then seemed close to their maximum potential, Cdiscount.com believed that the next improvement will be through deep learning. So, they have conducted this Kaggle competition.

## Source/Useful Links
* Source : https://www.kaggle.com/c/cdiscount-image-classification-challenge/overview

## Business problem and constraints
We need a model that automatically classifies the products based on their images.  The problem here is that, one product can have one or several images. The dataset Cdiscount.com is making available is having almost 9 million products and more than 15 million images at 180x180 resolution and more than 5000 categories that we need to classify the products in to. There are 3 levels of product classifications where there are groups inside the groups and sub-groups and so on. <br>
* <b> Objective :</b> Here our objective is to predict the probability of given image belonging to a particular product category as correctly as possible.<br>
* <b> Constraints : </b> There is some constraint regarding the latency as they cannot wait all day as many products are listed in their website on daily basis. So, we need a model that classifies the posted product with the given image in a reasonable timeframe.

## Deep Learning formulation of the business problem
Here given an image we need to perform a multi-class classification with 5000+ classes and predict the product category that the image belongs to as correctly as possible. The given below are the example images belonging to some classes.

<img width="571" alt="category_examples" src="https://user-images.githubusercontent.com/98249384/155836481-06ce0373-c520-49d6-9d89-e68710b9fa48.png">

## Dataset overview
The data is given in the form of BSON files, short for binary JSON files which are binary encoded serialization of the JSON-like documents, used with MongoDB. We need to read and process the BSON files to get the images.
File descriptions:
1. <b>	train.bson </b> : This file is huge with a size of 58.2GB and contains a list of 7Million dictionaries one per product. Each dictionary contains product id (key: _id), the category id of the product (key: category_id), and between 1-4 images, stored in a list (key: imgs). Each image list contains a single dictionary per image, which uses the format: {'picture': b'...binary string...'}. The binary string corresponds to a binary representation of the image in JPEG format. They have also provided an example of how to process this data.
2. <b>	test.bson </b> : This is also a huge file with a size of 14.5GB and contains a list of 1.7 million products in the same format as the train.bson, except there is no category_id included and this is the data that is used for evaluation for Kaggle and we need to create a submission file.
3. <b>	Category_names.csv </b> : This is a csv file which shows the hierarchy of the product classification. Each category_id has corresponding level1, level2 and level3 name, in French. The category_id corresponds to the category tree down to its lowest level. This hierarchical data may be useful, but it is not necessary for building models and making predictions.

## Notebooks Overview
### <a href="https://github.com/Rohan-Thoma/Ecommerce-product-image-classification/blob/master/pre-processing.ipynb"> pre-processing.ipynb </a> :
* As the dataset is huge, the data is provided in the compressed format in the BSON (Binary Java Script Object Notation) file whose file description is given above in the file description section.
* Here as there are 7 million images in total, we cannot extract all of them and it takes up a huge amount of space which is not available in a single box they take up a total of more than 700GB of space if extracted.
* Hence we extract only 1 million images by uniformly sampling across the whole dataset so that we can cover maximum amount of product categories with minimum number of images. Here we will skip 6 million images while parsing through the BSON file.

### <a href="https://github.com/Rohan-Thoma/Ecommerce-product-image-classification/blob/master/translation.ipynb"> translation.ipynb </a> :
* Here as CDiscount is a French Company , the product category names are given in french and we need to translate those names in to english for our easy interpretation.
* In this notebook , we have the code that will use google translate to convert the french class labels in to english class labels with a fair amount of accuracy.

### <a href="https://github.com/Rohan-Thoma/Ecommerce-product-image-classification/blob/master/EDA_cdiscount.ipynb"> EDA_cdiscount.ipynb </a> :
* This notebook contains the exploratory data analysis of the dataset, where we look at the class label hierarchy and also see the distribution of the various images amoung the different classes.
* We also explore the data imbalance degree which varies from one class level to another. We also plot the images to get an idea about of the easy and the hard examples.
* Here CD and the books will be the hard examples as they contain images printed on the cover which make the model confuse them with other products. Here the easy examples would be the products with white backgroud which highlight the product and are easy for the model to identify them.

### <a href="https://github.com/Rohan-Thoma/Ecommerce-product-image-classification/blob/master/Modelling_cdiscount.ipynb"> Modelling_cdiscount.ipynb </a> :
* Here we tried out various pretrained models with a variety of configurations.
* The various convolutional Neural networks like VGG-16,VGG-19,resnet-50, Inception-V3 have been tried out with various dataset configurations.
* Here the networks were trained both with imbalanced raw data and check for performance. As the performance was not good, data sampling was performed and then balancing the classes along with image augmentations gave comparatively good results.
* Here as there are 3 class levels with class level 3 with 5186 classes being the primary objective and due to the extreme skewness of the data, most of the classes are minority classes with less images. Here the classes with 1 image per class label were removed as they are very rare products and carry no bussiness value compared to the other product categories. The networks were trained to predict the 5070 level 3 class categories.
* Here for the better performance class level 2 which is the super class of the class level 3 is also taken as class labels and here as there are 481 classes and after checking the distribution of the class labels, it was found that less than 8% of the data belongs to classes with less than 1000 images which constitutes to 0.9 million images out of 1 million extracted images.Hence we have removed those classes and then the final remaining classes were 178.
* Here for the balancing the data, 1000 images from each class were randomly sampled and used for training various networks among which inceptionet_V3 gave better validation accuracy of 57%. 
* Here the score could be improved drastically if we tran with all the data which is close to 10 million + images.

### <a href="https://github.com/Rohan-Thoma/Ecommerce-product-image-classification/blob/master/Final_notebook.ipynb"> Final_notebook.ipynb </a> :
* This notebook contains the code for the entire pipeline for prediction of the category of any new image. Here we will resize the image to  the required size and then make the prediction for the top 4 classes of level 2 which are the most possible and likely classes for that product.

### <a href="https://github.com/Rohan-Thoma/Ecommerce-product-image-classification/blob/master/streamlit_app.py"> streamlit_app.ipynb </a> :
* This is the notebook containing the code regarding the deployed interactive app, which predicts the category of the image either uploaded by the user or it autonmatically scrapes a random image from the web and the user can see the performance of the model. The app also maintains the session registery of all the images given the user and the user can observe the model performance across various categories of images. Check out the app <a  href="https://share.streamlit.io/rohan-thoma/ecommerce-product-image-classification"> here. </a>

### Scores from the various deep learning networks are given below:
<table style="width:100%">
  <tr>
    <th>S.No</th>
    <th>Model</th>
    <th>Type of data</th>
   <th>Number of classes</th>
   <th>Test Accuracy</th>
  </tr>
  <tr>
   <td>1</td>
    <td>Baseline model</td>
    <td>on 1 million images </td>
   <td>5070 of class level 3</td>
   <td>44.82 %</td>
    </tr>
  <tr>
   <td>2</td>
   <td>Baseline model with class weights</td>
   <td>on 1 million images </td>
   <td>5070 of class level 3</td>
    <td>0 %</td>
  </tr>
  <tr>
   <td>3</td>
    <td>Resnet50 with fine tuning of last layers</td>
    <td>on 1 million images</td>
   <td>5070 of class level 3</td>
   <td>26.02 %</td>
  </tr>
  <tr>
   <td>4</td>
    <td>Resnet50 with fine tuning of all layers </td>
    <td>on 1 million images </td>
   <td>5070 of class level 3</td>
   <td>54.90 %</td>
  </tr>
 <tr>
  <td>5</td>
    <td>VGG-16 with fine tuning of last layers </td>
    <td>on 178k images with image augmentations</td>
  <td>178 of class level 2</td>
   <td>25.66 %</td>
  </tr>
  <tr>
   <td>6</td>
    <td>VGG-16 with fine tuning of all layers </td>
    <td>on 178k images with image augmentations</td>
   <td>178 of class level 2</td>
   <td>49.281 %</td>
  </tr>
 <tr>
   <td>7</td>
    <td>VGG-19 with fine tuning of all layers </td>
     <td>on 178k images with image augmentations</td>
   <td>178 of class level 2</td>
   <td>50.214 %</td>
  </tr>
 <tr>
   <td>8</td>
    <td>Resnet-50 with fine tuning of all layers </td>
     <td>on 178k images with image augmentations</td>
   <td>178 of class level 2</td>
   <td>49.674 %</td>
  </tr>
 <tr>
   <td>9</td>
    <td>Inceptionet-V3 with fine tuning of all layers </td>
     <td>on 178k images with image augmentations</td>
   <td>178 of class level 2</td>
   <td>58.313 %</td>
  </tr>



