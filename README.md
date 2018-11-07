# image_captioning-pytorch-en-de

Corwynne Leng: Image captioning model for Final Year project for Stellenbosch University.
This project does have some elements from the image captioning model at: https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning.
However, this model uses the Multi30k dataset instead of COCO, and can perform image captioning for English and German to reasonable success. It also implements its own version of beam search as well as a soft attention layer.
The following Python modules are needed: PyTorch, pandas, NLTK. I most likely missed out on a few modules that should be included.
The code definitely has not been refined and there are many redundant blocks of code or variables. But the main function does work.
You will need to extract the Flickr30 images into the data folder and rename the folder to 'flickr30k'.
Then perform the run the following files:
1) data_separation.py
2) vocabulary.py --lang=en  (you can use de to build the German vocabulary. The minimum amount of times a word has appeared is set to 1 but it can be set higher by changing the arguments)
3) datasetFormat.py (if you wish to generate your own csv files)
4) resize.py --mode=train (defaults to train but can resize validation images as well. No need to resize the test images)
5) trainattention.py
6) if you want to check the validation score, use attentionval.py
7) if you want to run on test set, use attentiontest.py
8) if you want to run on your own images use get_image.py --image=(Path of of image here), --lang=(which language you want to decode to)
Notes:
if you want to change model architectures, you can use the config.py file. CNN architectures that work are VGG-16 (vgg16) and ResNet-50 (resnet50). RNN uses LSTM (lstm) or GRU (gru).
This code can definitely be improved upon and again, have to reiterate, VERY, VERY ugly. 
