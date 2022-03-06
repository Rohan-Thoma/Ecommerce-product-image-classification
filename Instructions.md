## Introduction
Cdiscount.com is the largest non-food e-commerce company in France. The company has a wide variety of products which range from TVs to trampolines, the list of products is still rapidly growing. Cdiscount.com had over 30 million products up for sale. Ensuring that so many products are well classified is a challenging task. When all the well known machine learning models hit their maximum potential, next next improvement will be through deep learning. <br><br>


* The app given here is the demonstration of the case study on ecommerce image classification challenge whose code can be found at https://medium.com/@rohanvailalathoma/ecommerce-product-image-classification-for-cdiscount-com-ff2802d4636d <br>
* The entire code for the case study can be found at https://github.com/Rohan-Thoma/Ecommerce-product-image-classification <br> <br>

This classification challenge is difficult even with deep learning because if we look at the images of the products that we buy online, there are a wide variety of products and most of the images consists of considerable amount of background noise and other products which does not belong to that particular category. Such type of data is very different from the original imagenet data which implies that we need more data to train and mere fine tuning of the last few layers is not enough. So, here the the final model Inception_V3 is fine tuned with all the layers and with limited amount of data because of the resource constraints which cappped the performance of the model at 58% accuracy. This score could be improved with enough investment in the computational resources. The highest score obtained in the 1st place of the competition is 78% accuracy. <br><br> 

The details of the respective Kaggle competition : https://www.kaggle.com/c/cdiscount-image-classification-challenge/overview <br>

## Instructions
This app has 3 sections:
1. <b>Show instructions:</b> This section brings to this page which is the default home page of the page. This contains the intorduction to the problem and the instructions for the app. <br>
2. <b>Run the app:</b> This option runs the app. Here the user has to input image on which the classsfication can be performed. Here the input can be given in 2 ways:
    *  Upload an image yourself from the local storage
    *  The app gets a random product image from the web <br>
    *  After getting the image, the app will predict the 4 most likely categories that the image may belong to along with the respective probabilities. Here during the entire session, the user can get the predictions on the any number of images and the prediction history is shown in a tabular format.<br>
3. <b>Source code:</b> This section contains the entire source code of the app for quick reference. 
