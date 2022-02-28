import streamlit as st
from PIL import Image
import requests
import random
import pickle
import numpy as np
import io
from selenium import webdriver
import time
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.models import model_from_json

#This gets the chrome driver ready and it stays through out the session
if 'wd' not in st.session_state:
    
    #display the app status to the screen
    status_1 = st.markdown('Loading the webdriver...')
    
    #Load the webdriver from the disk
    st.session_state.wd=webdriver.Chrome(executable_path='cdiscount-image-classification-challenge\chromedriver.exe')
    
    #clear the status message displayed above
    status_1.empty()

#This loads the Inception_V3 network with trained weights and it stays through out the session
if 'model' not in st.session_state:
    
    #display the app status to the screen
    status_1 = st.markdown("Loading the model...")
    
    #get the model architecture first
    json_file = open('cdiscount-image-classification-challenge\model_design.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    #Load the saved weights
    st.session_state.model = model_from_json(loaded_model_json)
    st.session_state.model.load_weights("cdiscount-image-classification-challenge\model_weights.h5")
    
    #clear the app status displayed above
    status_1.empty()
   
#Initialize the list to save the image category predictions through out the session
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

#This is to load the class names for interpretation of the model predictions
if 'classes_list' not in st.session_state:
    with open('cdiscount-image-classification-challenge\class_names.pkl','rb') as pred_file:
        st.session_state.classes_list =  pickle.load(pred_file)
         
#This is load the markdown page for the entire home page
def get_file_content_as_string(path):
    with open(path,'r') as f:
        instructions=f.read()
    return instructions

#This is for the instructions home page
st.title('Ecommerce product image classification for CDiscount.com')
Main_image = st.image('cdiscount-image-classification-challenge\CDiscount.png')
readme_text=st.markdown(get_file_content_as_string('cdiscount-image-classification-challenge\Instructions.md'), unsafe_allow_html=True)

#This is for the side menu for selecting the sections of the app          
st.sidebar.markdown('# M E N U')
option = st.sidebar.selectbox('Choose the app mode',('Show instructions','Run the app', 'Source code'))

#function to show the developer information
def about():
    st.sidebar.markdown("# A B O U T")
    st.sidebar.image('cdiscount-image-classification-challenge\profile.png',width=180)
    st.sidebar.markdown("## Rohan Vailala Thoma")
    st.sidebar.markdown('* ####  Connect via [LinkedIn](https://in.linkedin.com/in/rohan-vailala-thoma)')
    st.sidebar.markdown('* ####  Connect via [Github](https://github.com/Rohan-Thoma)')
    st.sidebar.markdown('* ####  rohanvailalathoma@gmail.com')

#condition, if the user chooses the home page
if option == 'Show instructions':
    
    #alert options for further instructions to proceed
    success_text=st.sidebar.success('To continue, select "Run the app" ')
    warning_text=st.sidebar.warning('To see the code, go to "Source code"')
    
    #display the developer information
    about()

#condition if the user wishes to see the source code
if option == 'Source code':
    
    #erase the main page contents first
    Main_image.empty()
    readme_text.empty()
    
    #further instructions
    success_text=st.sidebar.success('To continue, select "Run the app" ')
    warning_text=st.sidebar.warning('Go to "Show instructions" to read more about the app')
    
    #display the whole sode stored in the text file
    text_file = open('cdiscount-image-classification-challenge/app_code.txt',mode='r')
    st.code(text_file.read())
    text_file.close()
    
    #display the developer information
    about()

#condition if the user chooses to run the app    
if option == 'Run the app':
    
    #erase the main page contents first
    Main_image.empty()
    readme_text.empty()
    
    #further instructions
    warning_text=st.sidebar.warning('Go to "Show instructions" to read more about the app')
    success_text=st.sidebar.success('To see the code, go to "Source code"')
    
    ##display the developer information
    about()
    
    #initialize the list to store the images collected through out the session
    if 'global_image_list' not in st.session_state:
        st.session_state.global_image_list=[]
    
    #define the fucntion to make the class label predictions
    def predict():
        
        #divider for the asthetics of the page
        st.write("-"*34)
        
        #compute the total number of images predicted by the user in the session and display it.
        number_of_images = len(st.session_state.global_image_list)
        st.write('#### Total products categorized : ', number_of_images)
        
        #make the predictions
        st.write('-'*34)
        if len(st.session_state.predictions) < number_of_images:
            
            #get the last image added to the list of images and pre-process it for prediction
            pred_image=st.session_state.global_image_list[-1]
            pred_image = pred_image.resize((128, 128))
            pred_image = np.expand_dims(pred_image, axis=0)

            #make the prediction
            pred = st.session_state.model.predict(pred_image,verbose=1)
            
            #make a list of the top 4 most likely categories for the product image
            pred_list=[]
            sorted_indices=np.argsort(pred[0])
            for h in range(4):
                pred_index=sorted_indices[-1-h]
                predicted_label=st.session_state.classes_list[pred_index]
                probability = np.round(pred[0][pred_index]*100,3)
                pred_list.append([predicted_label,probability])
            st.session_state.predictions.append(pred_list)
        
        #loop to display all the images categorized in the current session
        for i in range(number_of_images):
            
            #Display the images in the reverse order so that the latest predictions are at the top
            try:
                image_ = st.session_state.global_image_list[-1-i]
                preds_ = st.session_state.predictions[-1-i]
            except Exception as e:
                pass
            
            #This code is diaplay the predictions in the tabular format with 3 columns
            col1,col2,col3 = st.columns([1.5,2,1])
            
            #display the image in the 1st column
            with col1:
                if i==0:
                    st.write("### Image")
                    st.write("-"*40)
                st.image(image_,width=180)

            #display the class labels in the 2nd column
            with col2:
                if i==0:
                    st.write("### Product Category")
                    st.write("-"*40)
                    
                for g in range(4):
                    st.write('* ',preds_[g][0].upper())
            
            #display the probability scores in the 3rd column
            with col3:
                if i==0:
                    st.write("### Confidence")
                    st.write("-"*40)

                for g in range(4):
                    st.write('* ', preds_[g][1],' %')
                    
            st.write('-'*34)
            
    #this is the function to control the chrome browser with selenium to automatically search for the e-commerce images and get the urls.      
    def fetch_image_urls(query:str, max_links_to_fetch:int, wd:webdriver, sleep_between_interactions:int=5):
        
        def scroll_to_end(wd):
            wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(sleep_between_interactions)    
        
        # build the google query
        search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

        # load the page
        wd.get(search_url.format(q=query))

        image_urls = []
        image_count = 0
        results_start = 0
        while image_count < max_links_to_fetch:
            scroll_to_end(wd)

            # get all image thumbnail results
            thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
            number_results = len(thumbnail_results)
            
            print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")
            
            for img in thumbnail_results[results_start:number_results]:
                # try to click every thumbnail such that we can get the real image behind it
                try:
                    img.click()
                    time.sleep(sleep_between_interactions)
                except Exception:
                    continue

                # extract image urls    
                actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
                for actual_image in actual_images:
                    if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                        image_urls.append(actual_image.get_attribute('src'))

                image_count = len(image_urls)

                if len(image_urls) >= max_links_to_fetch:
                    print(f"Found: {len(image_urls)} image links, done!")
                    break
            else:
                print("Found:", len(image_urls), "image links, looking for more ...")
                time.sleep(30)
                return
                load_more_button = wd.find_element_by_css_selector(".mye4qd")
                if load_more_button:
                    wd.execute_script("document.querySelector('.mye4qd').click();")

            # move the result startpoint further down
            results_start = len(thumbnail_results)

        return image_urls    
        
    #ask for user preference
    st.markdown('### Choose your preferred method')
    genre = st.radio("",('Upload a product image yourself', 'Get a random product image from google automatically'))
    st.write("-"*34)
    
    #code get the uploaded image from the user
    if genre == 'Upload a product image yourself':
        
        #display the upload image interface
        uploaded_file = st.file_uploader("Choose an image")
        
        #check for errors in the file
        if uploaded_file is not None:
            
            try:
                #columns to get the tabular format with 2 columns
                col1,col2 = st.columns([1,1])
                
                #display the image in the left column
                with col1:
                    
                    #preprocess the image
                    image = Image.open(uploaded_file).convert('RGB')
                    image= image.resize((512,512))
                    
                    #add the image to the images list
                    st.session_state.global_image_list.append(image)
                    
                    #display the image to the screen
                    st.image(image,width=250)
                
                #display the feed-back further instructions in the right column    
                with col2:
                    
                    #further feedback and instructions
                    st.success('Successfully uploaded the image. Please see the category predictions below')
                    st.write("-"*34)
                    st.warning('To get a random image from the web, press the 2nd option above..!')
            
            #display the error message of the format is not an image format            
            except Exception as e:
                st.error('Please upload an image in jpg or png format..!')
        
        #Display the status while the model is predicting the class labels 
        status=st.text("getting the predictions..")
        
        #call the prediction method only if the image is uploaded or present
        if len(st.session_state.global_image_list) != 0:
            predict()
            
        #clear the status
        status.empty() 
      
    #This is the code to get the random image from google.                      
    if genre == 'Get a random product image from google automatically':
        
        #display the status to the screen
        status=st.text("Getting a random product image from google for you...")
        
        #official google API which is alternative to selenium but with a search limit
        #gis = GoogleImagesSearch('AIzaSyA3--ulfR-P4846NblrMI57BbbBEccwOjc', '23616888a5680a202')
        
        #get the class label names so that we can find the relevant images
        with open('cdiscount-image-classification-challenge\search_list.pkl','rb') as f:
            search_list = pickle.load(f)
        
        #randomly select a class label and create a search phrase to search the web
        search_word= 'buy '+ random.choice(search_list) + ' items online amazon'
        
        #This can be tweaked to get more than 1 image
        number=1
        
        #This is the code related to the Google API usage
        #_search_params = {
            #'q': search_word,
            #'num': number,
            #'fileType': 'jpg|gif|png',
            #'rights': 'cc_publicdomain|cc_attribute|cc_sharealike|cc_noncommercial|cc_nonderived'}
            
        #this will only search for images by calling the function which is defined above which outputs the image urls
        image_url=fetch_image_urls(search_word,1,st.session_state.wd)
        
        #This is the line of code for searching using the Google search API
        #gis.search(search_params=_search_params)
        
        #This is neccessary information headers to get the image, otherwise the websites will refuse to provide the images
        count=0
        hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                'Accept-Encoding': 'none',
                'Accept-Language': 'en-US,en;q=0.8',
                'Connection': 'keep-alive'}

        #Display the downloaded image in a tabular format
        col1,col2 = st.columns([1,1])
        
        #Display the image in the left column
        with col1:
            
            #this is to get the image from the given url
            for im in image_url:
                try:
                    
                    #get the images from the obtained urls
                    image_file = requests.get(im,headers=hdr).content
                    image_file = io.BytesIO(image_file) 
                    image = Image.open(image_file).convert('RGB')
                    
                    #some preprocessing that needs to be done    
                    image= image.resize((512,512))
                    
                    #add the image to the image list which contains all the images used in the session
                    st.session_state.global_image_list.append(image)
                    
                    #display the image to the screen
                    st.image(image,width=250)
                    count+=1

                #error messages in case is something goes wrong
                except Exception as e:
                    st.error("Could not get the image..!")
                    st.error("Please retry with another image..!")
                            
        #This is to display the further instructions in the right column
        with col2:
            
            #This is the button to try the prediction with another image
            if st.button('Try with another image',help='Literally the app goes to the google-images, searches for ecommerce products and gets the image'):
                genre='Get a random product image from google automatically'
            
            #These are the feedback and further instructions 
            st.success("Got a random product image from the web. Please see the predictions below..!")
            st.warning("Choose 'upload an image' to input your custom image..!")
             
        #clear the above status which says "getting the random image from the web..."
        status.empty()
        
        #Display the status while the model is predicting the class labels 
        status=st.text("getting the predictions..")
        
        #only predict if the image is uploaded by the user or if the image if present
        if len(st.session_state.global_image_list) != 0:
            predict()
        
        #clear the status
        status.empty()    
            
         
       

    

