from zhelper import *
from bayes.models import *
from bayes.explanations import BayesLocalExplanations
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t
import concurrent.futures
import itertools

def BayesShapExplainer (superpixels, compactness, bg, name, model, nPermute = 200, focus = True):
#Get image and segmentations
    image = get_image(name)
    image_transf = transform_and_segment(image, superpixels, compactness)

    #Get instances for training
    instance_one = image_transf['X']
    segments_one = image_transf['segments']
    xtrain = get_xtrain(image_transf['segments'])
    seg_num = len(np.unique(segments_one))
    image_greyscale = np.dot(image_transf['X'][..., :3], [0.2989, 0.5870, 0.1140])

    #Get core model for predictions
    basemodel = get_model(model)

    output = basemodel(image_transf['processed_img'])

    assert torch.argmax(output).item() == 355, f"{model} predicting wrong class"
    #Get a wrapped model (that takes in binary input)
    inference_model = get_model_for_shap(instance = instance_one, segments= segments_one, background = bg, model = basemodel,log_odd = True)

    #BayesSHAP
    exp_init = BayesLocalExplanations(training_data=xtrain,
                                    #training_data= xtrain,
                                    data="tabular",
                                    kernel="shap",
                                    categorical_features=np.arange(seg_num),
                                    verbose=True)

    rout = exp_init.explain(classifier_f= inference_model,
                            data=np.ones((1,seg_num))[0],
                            #label=np.argmax(prediction),
                            label = 355,
                            focus_sample= focus,
                            l2=False,
                            n_samples = nPermute,
                            feature_selection=False,
                            focus_sample_batch_size = 50,
                            focus_sample_initial_points = 50,
                            enumerate_initial=True,
                            #cred_width=1e-1
                            )
    
    
    return {
        'segmented_img': segments_one,
        'greyscale_img': image_greyscale,
        'explainer': rout,
        'seg_num': seg_num
    }



def BayesShapExplainerClass (superpixels, compactness, bg, name, model,classlabel,log_odd, nPermute = 200, focus = True):
#Get image and segmentations
    image = get_image(name)
    image_transf = transform_and_segment(image, superpixels, compactness)

    #Get instances for training
    instance_one = image_transf['X']
    segments_one = image_transf['segments']
    xtrain = get_xtrain(image_transf['segments'])
    seg_num = len(np.unique(segments_one))
    image_greyscale = np.dot(image_transf['X'][..., :3], [0.2989, 0.5870, 0.1140])

    #Get core model for predictions
    basemodel = get_model(model)

    output = basemodel(image_transf['processed_img'])

    #assert torch.argmax(output).item() == 355, f"{model} predicting wrong class"
    #Get a wrapped model (that takes in binary input)
    inference_model = get_model_for_shap(instance = instance_one, segments= segments_one, background = bg, model = basemodel, log_odd = log_odd)

    #BayesSHAP
    exp_init = BayesLocalExplanations(training_data=xtrain,
                                    #training_data= xtrain,
                                    data="tabular",
                                    kernel="shap",
                                    categorical_features=np.arange(seg_num),
                                    verbose=True)

    rout = exp_init.explain(classifier_f= inference_model,
                            data=np.ones((1,seg_num))[0],
                            #label=np.argmax(prediction),
                            label = classlabel,
                            focus_sample= focus,
                            l2=False,
                            n_samples = nPermute,
                            feature_selection=False,
                            focus_sample_batch_size = 50,
                            focus_sample_initial_points = 50,
                            enumerate_initial=True,
                            #cred_width=1e-1
                            )
    
    
    return {
        'segmented_img': segments_one,
        'greyscale_img': image_greyscale,
        'explainer': rout,
        'seg_num': seg_num,
        'original_img': instance_one
    }

# if __name__ == "__main__":
#     NAME = "Llama.png"
#     MODEL = "vgg16"
#     # BACKGROUND = 0
#     # SUPERPIXELS = 20
#     # COMPACTNESS = 1000
#     #head = True
#     RESULTFILE = f'result/result_final/experiment_{MODEL}.csv'
    
#     # list_of_dict = []
#     f = open(RESULTFILE, "w")
#     f.write(",image,superpixels,compactness,model,shap_value,background,predicted,predicted_variance,error term")
#     f.close()

#     def bayesWrapper (superpixels, compactness, background):
#         result =  BayesShapExplainer (superpixels, compactness, background, name = NAME, model = MODEL)
    
#         predicted,var = result['explainer']['blr'].predict(np.ones((1,result['seg_num'])))
#         new_data = {
#                 'image': [NAME] * result['seg_num'],
#                 'superpixels': [result['seg_num']] * result['seg_num'],
#                 'compactness': [compactness] * result['seg_num'],
#                 'model': [MODEL] * result['seg_num'],
#                 'shap_value': result['explainer']['coef'],
#                 'background': [background] * result['seg_num'],
#                 'predicted': [predicted.item()] * result['seg_num'],
#                 'predicted_variance': [var.item()] * result['seg_num'],
#                 'error term': [t.pdf(x =0, df = 200, scale = np.sqrt(result['explainer']['blr'].s_2))] * result['seg_num']
#                 }
#         #print(new_data)
#         print(f'{superpixels} pixels, {compactness} compactness, {background} background')
#         df = pd.DataFrame(data = new_data)
#         df.to_csv(RESULTFILE, mode = 'a', header = False)
#         #head = False

#     def helper(argument):
#         bayesWrapper(argument[0],argument[1],argument[2])

#     sp_list = (10,20,30,40,50)
#     cp_list = (10,30,50,100,1000)
#     bg_list = (0,'mean')

#     arg_list = [sp_list,cp_list,bg_list]
#     arg = list(itertools.product(*arg_list))
#     # print(arg)
#     # bayesWrapper(10,10,0)
#     with concurrent.futures.ThreadPoolExecutor(3) as executor:
#         executor.map(helper,arg)
#     # for background in bg_list:
#     #     for compact in cp_list:
#     #         with concurrent.futures.ThreadPoolExecutor() as executor:
#     #             executor.map (bayesWrapper, sp_list, repeat(compact), repeat(background))

    
#     # df = pd.DataFrame.from_dict(list_of_dict)
#     # print(df)
#     # df.to_csv(RESULTFILE, header = head)

#     #new_data.to_csv(RESULTFILE, mode = 'a', header = head)
#     # head = False
    

#     # result_image = fill_segmentation_test(result['explainer']['coef'], result['segmented_img'])
#     # plt.imshow (result['greyscale_img'], cmap = "gray", alpha = 0.5)
#     # plt.imshow (result_image, cmap = 'bwr', alpha = 0.6)
#     # plt.colorbar()
#     # plt.show()
