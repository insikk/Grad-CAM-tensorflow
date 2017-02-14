# Grad-CAM-tensorflow

This is tensorflow version of demo for Grad-CAM. I used vgg16 for demo because this model is very popular CNN model.
However grad-cam can be used with any CNN model. Just modify convolution layer in my demo code.

See [python notebook](https://github.com/insikk/Grad-CAM-tensorflow/blob/master/gradCAM_tensorflow_demo.ipynb) to see demo of this repository.
>To use the VGG networks in this demo, the npy files for [VGG16 NPY](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) has to be downloaded.

## [Origial Paper] Grad-CAM: Gradient-weighted Class Activation Mapping

**[Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization][7]**  
Ramprasaath R. Selvaraju, Abhishek Das, Ramakrishna Vedantam, Michael Cogswell, Devi Parikh, Dhruv Batra  
[https://arxiv.org/abs/1610.02391][7]

![Overview](http://i.imgur.com/JaGbdZ5.png)


## Acknowledgement

Thanks for the awesome machine learning commuity for providing many building blocks. 

### GradCAM implementation in caffe
[https://github.com/ramprs/grad-cam](https://github.com/ramprs/grad-cam)

### VGG16 implementation in tensorflow
[https://github.com/machrisaa/tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg)

### GradCAM implementation in keras, tensorflow
[https://github.com/jacobgil/keras-grad-cam](https://github.com/jacobgil/keras-grad-cam)

[https://github.com/Ankush96/grad-cam.tensorflow](https://github.com/Ankush96/grad-cam.tensorflow)

### Guided relu in tensorflow
[https://gist.github.com/falcondai/561d5eec7fed9ebf48751d124a77b087](https://gist.github.com/falcondai/561d5eec7fed9ebf48751d124a77b087)

### Guided backpropagation in tensorflow
[https://github.com/Lasagne/Recipes/blob/master/examples/Saliency%20Maps%20and%20Guided%20Backpropagation.ipynb](https://github.com/Lasagne/Recipes/blob/master/examples/Saliency%20Maps%20and%20Guided%20Backpropagation.ipynb)


[7]: https://arxiv.org/abs/1610.02391