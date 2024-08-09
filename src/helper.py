import os
import numpy as np
from PIL import Image
from torchvision import transforms,models
import torch
from skimage.segmentation import slic
from math import log
from copy import deepcopy


def get_image(name):
    image_path = os.path.join("./data", name)
    with open(os.path.abspath(image_path), 'rb') as f:
        with Image.open(f) as img:
            image = img.convert('RGB') 
    
    return image

def image_preprocessor():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]) 
    return transf

def transform_and_segment(image,superpixels, compactness):
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])
    transf_image = transf(image)
    segments = slic(transf_image, n_segments=superpixels, compactness=compactness, sigma=1)
    
    # fig = plt.figure(figsize=(10, 7)) 
    # fig.add_subplot(1,2,1)
    # plt.imshow(transf_image)
    # plt.axis('off')
    
    # fig.add_subplot(1,2,2)
    # plt.imshow(segments)
    # plt.axis('off')

    img = np.array(transf_image)
    seg = np.array(segments)

    preprocessor = image_preprocessor()

    output = {
        "X": img,
        "segments": seg,
        "processed_img": preprocessor(img).reshape((-1,3,224,224))
    }
    return output

def get_model(name):
    if name == 'vgg16':
        basemodel = models.vgg16(weights ='DEFAULT')
    elif name == 'resnet50':
        basemodel = models.resnet50(weights ='DEFAULT')
    elif name == 'resnet152':
        basemodel = models.resnet152(weights ='DEFAULT')
    elif name == 'googlenet':
        basemodel = models.googlenet(weights ='DEFAULT')
    elif name == 'visiontransformer':
        basemodel = models.vit_b_16(weights = 'DEFAULT')
    elif name == 'alexnet':
        basemodel = models.alexnet(weights = 'DEFAULT')
    else:
        print("vgg16 is chosen by default")
        basemodel = models.vgg16(weights ='DEFAULT')
    basemodel.eval()
    return basemodel

def fill_segmentation_test(values, segmentation):
    out = np.zeros(segmentation.shape)
    for i in range(len(values)):
        out[segmentation == (i+1)] = values[i]
    return out

def get_model_for_shap(instance, segments,model,log_odd = True, background = 0):
    def model_for_shap(data):
        preprocessor = image_preprocessor()
        perturbed_images = []

        if type(background) == int:
            for img_index in range(data.shape[0]):
                perturbed_image = deepcopy(instance)
                for i, is_on in enumerate(data[img_index]):
                    if is_on == 0:
                        perturbed_image[segments == (i+1), 0] = background
                        perturbed_image[segments == (i+1), 1] = background
                        perturbed_image[segments == (i+1), 2] = background
                
                #Preprocess (Change Dimension + Normalize)
                preprocessed_image = preprocessor(perturbed_image)

                #Post transposed, this can be put into the model
                perturbed_images.append(preprocessed_image)

        elif background == "mean":
            for img_index in range(data.shape[0]):
                perturbed_image = deepcopy(instance)
                for i, is_on in enumerate(data[img_index]):
                    if is_on == 0:
                        perturbed_image[segments == (i+1), 0] = perturbed_image[segments == (i+1), 0].mean()
                        perturbed_image[segments == (i+1), 1] = perturbed_image[segments == (i+1), 1].mean()
                        perturbed_image[segments == (i+1), 2] = perturbed_image[segments == (i+1), 2].mean()
                
                #Preprocess (Change Dimension + Normalize)
                preprocessed_image = preprocessor(perturbed_image)

                #Post transposed, this can be put into the model
                perturbed_images.append(preprocessed_image)

        elif background == "median":
            for img_index in range(data.shape[0]):
                perturbed_image = deepcopy(instance)
                for i, is_on in enumerate(data[img_index]):
                    if is_on == 0:
                        perturbed_image[segments == (i+1), 0] = np.median(perturbed_image[segments == (i+1), 0])
                        perturbed_image[segments == (i+1), 1] = np.median(perturbed_image[segments == (i+1), 1])
                        perturbed_image[segments == (i+1), 2] = np.median(perturbed_image[segments == (i+1), 2])
                
                #Preprocess (Change Dimension + Normalize)
                preprocessed_image = preprocessor(perturbed_image)

                #Post transposed, this can be put into the model
                perturbed_images.append(preprocessed_image)

        #Here we got all images, we need to stack them together
        perturbed_images = torch.stack(perturbed_images)
        
        #Feed into the model
        predictions = model(perturbed_images)

        #Convert results to numpy array
        softmax = torch.nn.Softmax(dim=1)
        predictions = softmax(predictions).detach().numpy()
        if log_odd:
            log_odd_transform = lambda x: log(x/(1-x))
            log_odd_function = np.vectorize(log_odd_transform)

            predictions = log_odd_function(predictions)
        
        #predictions = predictions.detach().numpy()

        
        #predictions = predictions[:,label]
        return predictions
    return model_for_shap