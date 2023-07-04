import cv2
import numpy as np
import streamlit as st
from PIL import Image
import hsr_model
import model_config

def resizing(img):
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    if w > 1000:
        new_w = 1000
        ar = w / h
        new_h = int(new_w/ar)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

def main():
    st.title("Handwritting Recognition")
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        # check if the "Predict" button has been clicked
        if image is not None and st.button("Scan"):
            image = resizing(image)
            predicted_text = hsr_model.predict(image)
            st.write("Predicted text: " + predicted_text)
            # st.write(image.shape)
            
if __name__ == "__main__":
    main()
