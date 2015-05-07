import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '/usr/prakt/p048/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
# sys.path.insert(0, caffe_root + '/python')
sys.path.append("/usr/prakt/p048/selective_search_ijcv_with_python")
sys.path.append("/usr/prakt/p048/caffe/python")

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

IMAGE_FILE1 = '/usr/prakt/p048/Downloads/ex2_images/IMG_1.jpg'
IMAGE_FILE2 = '/usr/prakt/p048/Downloads/ex2_images/IMG_2.jpg'
IMAGE_FILE3 = '/usr/prakt/p048/Downloads/ex2_images/IMG_3.jpg'
IMAGE_FILE4 = '/usr/prakt/p048/Downloads/ex2_images/IMG_4.jpg'
IMAGE_FILE5 = '/usr/prakt/p048/Downloads/ex2_images/IMG_5.jpg'
IMAGE_FILE6 = '/usr/prakt/p048/Downloads/ex2_images/IMG_6.jpg'
IMAGE_FILE7 = '/usr/prakt/p048/Downloads/ex2_images/IMG_7.jpg'
IMAGE_FILE8 = '/usr/prakt/p048/Downloads/ex2_images/IMG_8.jpg'

caffe.set_mode_cpu()

net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))


input_images = [caffe.io.load_image(IMAGE_FILE1), caffe.io.load_image(IMAGE_FILE2), caffe.io.load_image(IMAGE_FILE3), caffe.io.load_image(IMAGE_FILE4), caffe.io.load_image(IMAGE_FILE5), caffe.io.load_image(IMAGE_FILE6), caffe.io.load_image(IMAGE_FILE7), caffe.io.load_image(IMAGE_FILE8)]



caffe.set_mode_gpu()


# # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
# transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# transformer.set_transpose('data', (2,0,1))
# transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
# transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
# transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


prediction = net.predict(input_images)


prediction
# load labels
imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

print 'Image 1:'
print 'predicted class name:', labels[prediction[0].argmax()]
print 'predicted class probability:', prediction[0][prediction[0].argmax()]

print 'Image 2:'
print 'predicted class name:', labels[prediction[1].argmax()]
print 'predicted class probability:', prediction[1][prediction[0].argmax()]

print 'Image 3:'
print 'predicted class name:', labels[prediction[2].argmax()]
print 'predicted class probability:', prediction[2][prediction[0].argmax()]

print 'Image 4:'
print 'predicted class name:', labels[prediction[3].argmax()]
print 'predicted class probability:', prediction[3][prediction[0].argmax()]

print 'Image 5:'
print 'predicted class name:', labels[prediction[4].argmax()]
print 'predicted class probability:', prediction[4][prediction[0].argmax()]

print 'Image 6:'
print 'predicted class name:', labels[prediction[5].argmax()]
print 'predicted class probability:', prediction[5][prediction[0].argmax()]

print 'Image 7:'
print 'predicted class name:', labels[prediction[6].argmax()]
print 'predicted class probability:', prediction[6][prediction[0].argmax()]

print 'Image 8:'
print 'predicted class name:', labels[prediction[7].argmax()]
print 'predicted class probability:', prediction[7][prediction[0].argmax()]
