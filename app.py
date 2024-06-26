import streamlit as st 
from helper import (
    get_text_from_image , 
    get_speech_from_text
)

def project() : 
    '''
    The main function that contains the logic for the project.

    Args:

    Returns:
    '''

    image = st.file_uploader('Upload an image' , type = ['jpg' , 'jpeg' , 'png' , 'webp'])

    if image :

        with st.spinner('Extracting and Analysing Text from the Image, Please Hang Tight !!') : 

            text = get_text_from_image(image)

        st.write(text)

        with st.spinner('Getting Text to Speech') : 
            get_speech_from_text(text)

        st.audio('audio.wav' , format = 'audio/wav')

def report() : pass
def synopsis() : pass
def teams() : pass


option = st.sidebar.selectbox(
    'Go to' , 
    [
        'Project' ,
        'Report' , 
        'Synopsis' , 
        'Teams'
    ])

if option == 'Project' : project()
elif option == 'Report' : report()
elif option == 'Synopsis' : synopsis()
elif option == 'Teams' : teams()
