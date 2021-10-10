Testing out [detecto](https://github.com/alankbi/detecto) - "a Python package that allows you to build fully-functioning computer vision and object detection models with just 5 lines of code"

Disclaimer: In the title, by fast - I meant easy training and no frills data set up, NOT low latency

# Installation:

pip install -r requirements.txt


# Steps:

python data_prep.py  
python train.py   
python inference.py  


# Notes:

Model architecture - Faster R-CNN Resnet 50    
Data - [Grocery Dataset](https://github.com/gulvarol/grocerydataset)  
