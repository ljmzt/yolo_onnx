# Yolo_onnx

## Purpose
This is to learn how to convert models between Onnx and PyTorch. The example I chose is the tiny yolo v2 model from the [onnx model zoo](https://github.com/onnx/models/tree/main/validated/vision/object_detection_segmentation/tiny-yolov2). 

## Files
yolomodel.py: Contains the model class that constrcuts the tiny yolo model in pytorch and loads the onnx model. This is the most important file in this exercise.

tinyyolov2-7.onnx: The onnx model downloaded from the link above.

converter.py: A helper class that turns the output into bounding boxes, which includes a nonmax suppresion.

mytransforms.py: Includes the PasteImage transform, which pastes an image of size H * W image inside an H' * H' grey image, where H' is the larger between H and W. See below for explanation.

mydisplay.py: Just to double check the result for a single frame, not needed in actual deployment.

put_all_together.ipynb: A notebook when we put everything together. It captures the input from your camera and puts some bounding boxes for things it recognizes.

## Things to pay attention to
1. One fun thing for this onnx model is that it has a maxpooling layer with kerenl_size 2, stride 1 and maintains the input shape (13x13). It is actually impossible to do this using the default maxpooling layer of pytorch. My way to get around it is to first pad the input, then do the regular maxpooling. See MaxPoolSameSize class in the yolomodel.py.

2. It's also important to note that the definition of momentum between onnx and pytorch are opposite to each other in the batchnorm layer. The tinyyolov2 model has a momentum of 1.0 in the onnx implementation, which translates to 0.0 in pytorch.

3. In terms of the input image size, I initially think it won't matter as only convolution layers are used in the model. However, when I test it, using arbitrary input size significantly deteriorates the output. I believe this is because the anchor boxes are tied to the original size. So my way to get around it is to insert the input figure (HxW) into a grey square image, whose height H' is the longer one between H and W, using the PasteImage transform in mytransforms.py. Then do the regular resizing. Some extra post-processing is needed for the output bounding box to make things self-consistent, and please refers put_all_together.ipynb.

## Libraries
python=3.9.18  
pytorch=1.12.1  
torchvision=0.13.1  
pillow=8.1.2  
onnx=1.14.0  
opencv=4.5.1
