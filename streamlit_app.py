import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image


def model(text_picture):
    img = np.array(text_picture.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    if w > 1000:
        new_w = 1000
        ar = w/h
        new_h = int(new_w/ar)
        img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA)

    #plt.imshow(img)
    def thresholding(image):
        img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(img_gray,80,255,cv2.THRESH_BINARY_INV)
        plt.imshow(thresh, cmap='gray')
        return thresh

    thresh_img = thresholding(img)
    #dilation
    kernel = np.ones((3,85), np.uint8)
    dilated = cv2.dilate(thresh_img, kernel, iterations = 1)
    plt.imshow(dilated, cmap='gray')
    (contours, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[1]) # (x, y, w, h)
    
    img2 = img.copy()
    for ctr in sorted_contours_lines:
        x,y,w,h = cv2.boundingRect(ctr)
        cv2.rectangle(img2, (x,y), (x+w, y+h), (40, 100, 250), 2)
    
    #dilation
    kernel = np.ones((3,15), np.uint8)
    dilated2 = cv2.dilate(thresh_img, kernel, iterations = 1)
    plt.imshow(dilated2, cmap='gray')

    img3 = img.copy()
    words_list = []
    for line in sorted_contours_lines:
        # roi of each line
        x, y, w, h = cv2.boundingRect(line)
        roi_line = dilated2[y:y+w, x:x+w]
        # draw contours on each word
        (cnt, heirarchy) = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contour_words = sorted(cnt, key=lambda cntr : cv2.boundingRect(cntr)[0])

        for word in sorted_contour_words:
            if cv2.contourArea(word) < 400:
                continue
            x2, y2, w2, h2 = cv2.boundingRect(word)
            words_list.append([x+x2, y+y2, x+x2+w2, y+y2+h2])
            cv2.rectangle(img3, (x+x2, y+y2), (x+x2+w2, y+y2+h2), (255,255,100),2)

    ninth_word = words_list[8]
    roi_9 = img[ninth_word[1]:ninth_word[3], ninth_word[0]:ninth_word[2]]
    return roi_9


def main():
    st.title("Handwritting Recognition")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    if st.button("Scan"):  # check if the "Predict" button has been clicked
        value = model(image)
        st.write("Result:")
        st.image(value, use_column_width=True)

if __name__ == "__main__":
    main()
