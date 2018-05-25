# Grad-CAM-tensorflow

**NOTE: There is another awesome visualization of CNN called [CNN-Fixations](https://github.com/val-iisc/cnn-fixations), which involvs only forward pass. Demo code is available for Caffe and Tensorflow ResNet, Vgg. Please check it out.**

This is tensorflow version of demo for Grad-CAM. I used ResNet-v1-101, ResNet-v1-50, and vgg16 for demo because this models are very popular CNN model.
However grad-cam can be used with any other CNN models. Just modify convolution layer in my demo code.

![Preview](https://github.com/insikk/Grad-CAM-tensorflow/blob/master/image_preview.png?raw=true)

See [python notebook](https://github.com/insikk/Grad-CAM-tensorflow/blob/master/gradCAM_tensorflow_demo.ipynb) to see demo of this repository.
>To use VGG networks in this demo, the npy files for [VGG16 NPY](ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy) has to be downloaded.

>To use ResNet-v1-50 or ResNet-v1-101, download weight from https://github.com/tensorflow/models/tree/master/research/slim


**Any Contributions are Welcome**


## [Origial Paper] Grad-CAM: Gradient-weighted Class Activation Mapping

**[Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization][7]**  
Ramprasaath R. Selvaraju, Abhishek Das, Ramakrishna Vedantam, Michael Cogswell, Devi Parikh, Dhruv Batra  
[https://arxiv.org/abs/1610.02391][7]

![Overview](http://i.imgur.com/JaGbdZ5.png)

# Requirements

* GPU Memory: 6GB or higher to run VGG16, and ResNet101 (You may able to run ResNet50 with less than 6GB)

# Setup

```
export PYTHONPATH=$PYTHONPATH:`pwd`/slim
```

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
