# DobaraMatPuchana

## **VisioVoice** : Assistant for blind

This voice assistant is designed with a primary focus on improving the quality of life for individuals with visual impairments. It offers a range of customized functionalities and employs speech-to-text technology to assist users in accessing essential features. In essence, it serves as a dedicated chatbot tailored to meet the unique needs of the visually impaired community.

## Fetaures which make "VisioVoice" unique:

The system comprises wearable headphones connected to a Logitech webcam, which is in turn interfaced with a Raspberry Pi. It operates by interpreting user input in the form of speech and, in response, provides spoken feedback while executing various tasks.

Key Features:

1. **Description**:
  
   1. VisioVoice provides a concise summary of the immediate surroundings.
   2. Under the category of "Road Conditions," VisioVoice offers a comprehensive assessment to assist visually impaired users.
   3. VisioVoice is capable of identifying and specifying familiar locations, such as classrooms, kitchens, and bedrooms.
   4. The system responds by indicating the quantity of individuals, objects, and more within the webcam's frame.

2. **Find**:

    1. VisioVoice resonds to commands like *find my purse?*, *check if my watch is in this room?* depending upon whether @Entity is present in the frame of the camers
  
3. **Read**:

    1. VisioVoice also detects text from images and reads it loud.
    2. As a further application it can summarize articles from newspapers. 
    
4. **Fill forms**
    
    1. VisioVoice also reads out forms (majorly applicable for bank purposes)
    
5. **Mobile Interactions**

    1. It can read out notifications from mobile and as a further application respond to messages, emails, calender, etc
    
6. **Add ons**

    1. VisioVoice serves the basic features of a chat bot i.e. responds to question including time, lighting conditions, basic wh questions, etc.
    
    
Tech- stacks used:

    1. SpeechRecognizer, Google API for speech to text conversion
    
    2. Python Text to Speech https://pypi.org/project/pyttsx3/
    
    3. Object Recogniiton using *COCO Dataset*
    
    4. Google Cloud Vision API 
    
    5. Dialogflow
            
 Install Dependencies
 Download the credential for Google Vision API.
 Copy and paste the files at thr required position.
 
 ```
    git clone https://github.com/Sumanthkumar2002/-VisioVoice-Assistant-for-blind.git
    cd DobaraMatPuchana
    pip install -r requirements.txt
 ```
 
 Run Code:
 
 ```
 python main.py
 ```
