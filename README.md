# Traffic-Sign-Classification
A deep-learning traffic sign detection and recognition project, using convolutional neural networks and vision transformers, with PyTorch over the [GTSRB - German Traffic Sign](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?resource=download). 

There are three main components currently planned for this project:

1) Traffic Sign Classification - in development <br />
A model that, given any RGB image of a traffic sign, can classify it into 1 of 43 possible classes.

    The [classification demo notebook](https://github.com/alvaroqsaldanha/Traffic-Sign-Classification/blob/main/demo.ipynb) showcases examples of predictions made with       the CNN and the Vision Transformer, and provides a walkthrough of multiple development steps, including data preparation, training, and testing of the models. A section comparing both models is also being developed.
    
    Some examples of predictions using the CNN:
    
    ![Example of predictions](https://github.com/alvaroqsaldanha/Traffic-Sign-Classification/blob/main/DataProfiling/templateimg.PNG)
    
    CNN vs Current Vision Transformer Model Comparison - Accuracy and Loss over Epochs:
    
    ![Comparison Between Models](https://github.com/alvaroqsaldanha/Traffic-Sign-Classification/blob/main/DataProfiling/templateimgComparison.PNG)

2) Traffic Sign Detection - in development <br /> 
Another model that, given an image, detects the presence of a traffic sign and give its pixel coordinates, allowing for posterior classification. Currently, I plan to use a Yolo-based approach to this section.

3) Web-app <br />
Deploy a web-app, using Flask or Django, that allows the user to upload an image and run the models.

## About the data:

The GTSRB dataset (German Traffic Sign Recognition Benchmark) has images pertaining to 43 classes of traffic signs, containing 39,209 train examples and 12,630 test ones. It was provided by the [Institut für Neuroinformatik](https://benchmark.ini.rub.de/?section=gtsrb&subsection=news) in 2011. In the [classification demo notebook](https://github.com/alvaroqsaldanha/Traffic-Sign-Classification/blob/main/demo.ipynb), data profiling is done to help visualize the dataset.

## Depedencies:

> pandas <br>
> numpy <br>
> PyTorch <br>
> matplotlib <br>
> Pillow <br>
> scikit-learn <br>
> pickle <br>
> torchvision <br>
> opencv <br>




