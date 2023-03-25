Detection of Face Mask Using Computer Vision and Convolutional Neural Network (CNN).
Since the beginning of the new Covid19 disease outbreak, public use of wearing masks has 
become very important everywhere in the whole world, this project proposes a method to
automate the face mask detection process for hospitals so the computer vision can detect if a person 
wear a face mask or not and emits red light if someone doesn't wear face mask.

We build five CNN models then we train them with 2100 images and test them with 900 images. We use the same preprocessing method for all models. After training and validation, we saved the model and tested it on the test images on another 100 test images and the result is shown in the table.

After this comparison and choosing the best model, the result shown below. 

![image](https://user-images.githubusercontent.com/84765301/220972028-5c4fcfbe-0f73-459d-91d6-dea30f47d8e2.png) ![image](https://user-images.githubusercontent.com/84765301/220972062-da3d254d-88f5-45af-a213-cc2f34d0d84f.png)

Then we developed it more to also detected the veil as shown:

![image](https://user-images.githubusercontent.com/84765301/221044366-7438109c-df31-419d-b71a-0fe4885fb198.png)
