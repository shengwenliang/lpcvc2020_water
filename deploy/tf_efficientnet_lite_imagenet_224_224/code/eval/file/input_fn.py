import cv2
from sklearn import preprocessing

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

MEANS = [_B_MEAN,_G_MEAN,_R_MEAN]

IMAGE_SIZE = 224
CROP_PADDING = 32
MEAN_RGB = [127.0, 127.0 , 127.0]
STDDEV_RGB = [128.0, 128.0, 128.0]

def efficient_padding_crop_resize(image, image_size = IMAGE_SIZE, crop_padding=32):

  h, w = image.shape[:2]

  padded_center_crop_size = int((image_size / (image_size + crop_padding)) * min(h, w)
                                      )
  offset_height = ((h - padded_center_crop_size) + 1) // 2
  offset_width = ((w - padded_center_crop_size) + 1) // 2

  image_crop = image[offset_height: padded_center_crop_size + offset_height, offset_width: padded_center_crop_size + offset_width,:]
  image = cv2.resize(image_crop, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
  image = image - MEAN_RGB
  image = image / STDDEV_RGB
  return image





def resize_shortest_edge(image, size):
  H, W = image.shape[:2]
  if H >= W:
    nW = size
    nH = int(float(H)/W * size)
  else:
    nH = size
    nW = int(float(W)/H * size)
  return cv2.resize(image,(nW,nH))

def mean_image_subtraction(image, means):
  B, G, R = cv2.split(image)
  B = B - means[0]
  G = G - means[1]
  R = R - means[2]
  image = cv2.merge([R, G, B])
  return image

def BGR2RGB(image):
  B, G, R = cv2.split(image)
  image = cv2.merge([R, G, B])
  return image

def central_crop(image, crop_height, crop_width):
  image_height = image.shape[0]
  image_width = image.shape[1]
  offset_height = (image_height - crop_height) // 2
  offset_width = (image_width - crop_width) // 2
  return image[offset_height:offset_height + crop_height, offset_width:
               offset_width + crop_width, :]

def normalize(image):
  image=image/256.0
  image=image-0.5
  image=image*2
  return image


#def preprocess_fn(image, crop_height, crop_width):
#    image = resize_shortest_edge(image, 256)
#    image = mean_image_subtraction(image, MEANS)
#    image = central_crop(image, crop_height, crop_width)
#    return image 

eval_batch_size = 1
def eval_input(iter, eval_image_dir, eval_image_list, class_num):
    images = []
    labels = []
    line = open(eval_image_list).readlines()
    for index in range(0, eval_batch_size):
        curline = line[iter * eval_batch_size + index]
        [image_name, label_id] = curline.split(' ')
        image = cv2.imread(eval_image_dir + image_name)
        image = BGR2RGB(image)
        image = efficient_padding_crop_resize(image)
        images.append(image)
        labels.append(int(label_id))
    lb = preprocessing.LabelBinarizer()
    lb.fit(range(0, class_num))
    labels = lb.transform(labels)
    return {"input": images, "labels": labels}


calib_image_dir = "/workspace/CBIR/ILSVRC2012_img_val/"
calib_image_list = "images/tf_calib.txt"
calib_batch_size = 10
def calib_input(iter):
    images = []
    line = open(calib_image_list).readlines()
    for index in range(0, calib_batch_size):
        curline = line[iter * calib_batch_size + index]
        calib_image_name = curline.strip()
        image = cv2.imread(calib_image_dir + calib_image_name)  
        image = BGR2RGB(image)
        image = efficient_padding_crop_resize(image)
        images.append(image)
    return {"images": images}

