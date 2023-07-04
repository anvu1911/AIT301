import cv2
import numpy as np
import streamlit as st
from PIL import Image


def thresholding(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY_INV)
    return thresh


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


def dilating(img):
    kernel = np.ones((5, 100), np.uint8)
    dilated = cv2.dilate(img, kernel, iterations=1)
    return dilated


def segmenting(img, dilated):
    (contours, hierarchy) = cv2.findContours(
        dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(
        contours, key=lambda ctr: cv2.boundingRect(ctr)[1])  # (x, y of top left corner, w, h)

    img2 = img.copy()

    for ctr in sorted_contours_lines:
        x, y, w, h = cv2.boundingRect(ctr)

        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        cropped_image = img2[y1:y2, x1:x2]
        st.image(cropped_image, use_column_width=True)


def segmentation(text_picture):
    img = resizing(text_picture)

    thresh_img = thresholding(img)

    dilated_img = dilating(thresh_img)

    st.header("Dilated lines")
    st.image(dilated_img, use_column_width=True)

    st.header("Segmented regions")
    segmenting(img, dilated_img)


def main():
    st.title("Handwritting Recognition")
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        # check if the "Predict" button has been clicked
        if image is not None and st.button("Scan"):
            segmentation(image)



if __name__ == "__main__":
    main()
