<h1>AI Model Training & Deployment Procedures</h1>

1. **Obtain Images:** First, obtain varied images of toy cars in the desired location
2. **Labeling:** Label these images utilizing the Label Studios Software. 
3. **Augmentation:** Run these images through the Augments_9PerImage to increase the variety. If more images are desired, the number of augments per image can be changed by modifying the num_augmentations variable inside the file. 
4. **Upload to Colab:** Then, upload these new images and labels to Google Colab. If images are not uploading, utilize tools such as XnConvert to reduce the quality and size of images
5. **Train Model:** Upload directly to open source Google Colab(Train_YOLO_Models.ipynb) and follow the instructions to train the AI model
6. **Transfer Model:** Upon concluding training, move the downloaded file with the AI model onto the Raspberry Pi
7. **Setup Environment:** Establish a venv and install necessary packages on Raspberry Pi (install Ultralytics NCNN)
8. **Import Scripts:** Import index1.html and yolo_web_testing_headless_with_live.py to Raspberry Pi
9. **Run Server:** Start running server using: 
   ```bash
   python3 yolo_web_testing_headless_with_live.py --model=my_model4_ncnn_model --source=usb0 --resolution=1280x640 --headless
