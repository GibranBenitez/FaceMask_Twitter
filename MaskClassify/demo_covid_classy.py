from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import shutil
import pdb
import sys

from utils.extra_loaders import demo_loader
from modelos.iresnet import iresnet50
from modelos.iresnet_face import iresnet50
# os.environ["CUDA_VISIBLE_DEVICES"]="0"



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, use_pretrained=False, feature_extract=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "iresnet50":
        """ iresnet50
        """
        model_ft = iresnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "iresnet50":
        """ iresnet50 with face dataset
        """
        model_ft = iresnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "MobileNetV3-L":
        """ MobileNetV3 Large
        """
        model_ft = models.mobilenet_v3_large(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[3].in_features
        model_ft.classifier[3] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "ResNeXt-50":
        """ ResNeXt-50 32x4d
        """
        model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "efficientnet":
        """ EfficientNet B0 
        """
        model_ft = models.efficientnet_b0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "convnext-tiny":
        """ ConvNeXt Tiny
        """
        model_ft = models.convnext_tiny(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[2].in_features
        model_ft.classifier[2] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "convnext-small":
        """ ConvNeXt Small
        """
        model_ft = models.convnext_small(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[2].in_features
        model_ft.classifier[2] = nn.Linear(num_ftrs, num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft


def mask_classy(out_dir, in_dir, word, dates, weights_name, batch_size=32, th=0.87):

    if dates == None:
        data_dir = in_dir
        result_dir = out_dir
    else:
        data_dir = os.path.join(in_dir, word, dates)
        result_dir = os.path.join(out_dir, dates)

    txt_file = os.path.join(out_dir, '{}_{}_classy.txt'.format(dates,word))

    if dates == None:
        print('######### Results for {} ############'.format(in_dir))
        file.write('######### Results for {} ############\n'.format(in_dir))
    else:
        print('######### Results for {} {} ############'.format(word, dates))
        file.write('######### Results for {} {} ############\n'.format(word, dates))

    os.makedirs(out_dir, exist_ok=True)
    file = open(txt_file, "w+")

    out_model_path = './weights/{}.pth'.format(weights_name)
    print(out_model_path)
    args_model = weights_name.split('_')

    # Number of classes in the dataset
    class_names = ['Cloth', 'None', 'Respirator', 'Surgical', 'Valve']
    num_classes = len(class_names)
    # Adding Others label for demo
    class_names = class_names + ['Others']

    # Models to choose from [resnet50, resnet101, convnext-tiny   efficientnet   ResNeXt-50   MobileNetV3-L]
    model_name = args_model[0]
    if args_model[-1] == "face":
        data_std = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    else:
        data_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input_size = int(args_model[1][1:])
    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file.write('\nModel: {}, input_size: {}, pretrain_dataset: {}, Th: {}\n'.format(model_name, input_size, args_model[-1], th))

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(data_std[0], data_std[1])    ]),
    }

    print("Initializing Datasets and Dataloaders...")

    image_dataset = demo_loader(data_dir, data_transforms['test'])
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    dataset_sizes = len(image_dataset) 
    print(dataset_sizes)
    print(class_names)
    file.write('Dataset size: {}, num_classes: {}, top_path: {}\n'.format(dataset_sizes, len(class_names), in_dir))
    
    # Initialize the model for this run
    model_ft = initialize_model(model_name, num_classes)
    # Print the model we just instantiated
    print("Initializing {} model".format(model_name))
    # Send the model to GPU
    model_ft = model_ft.to(device)
    # Train and evaluate
    model_ft.load_state_dict(torch.load(out_model_path))
    print('Loaded {} model'.format(out_model_path))
    file.write('\nLoaded {} model\n'.format(out_model_path))

    since = time.time()
    probsS_history = []
    probs_history = []
    preds_history = []
    path_history = []
    model_ft.eval()
    norSoft = nn.Softmax()
    i = 0

    for fol_ in class_names:
        os.makedirs(os.path.join(result_dir, fol_), exist_ok=True)

    for inputs, inputs_paths in dataloader:
        i += 1
        inputs = inputs.to(device)

        outputs = model_ft(inputs)

        _, preds = torch.max(outputs, 1)
        probsM, _ = torch.max(norSoft(outputs), 1)
        img_preds = preds.tolist()
        img_probs = probsM.tolist()
        for idl, prob in enumerate(img_probs):
            if prob < th:
                img_preds[idl] = -1
            print_str = "  {}/{}, {}, {}  ,{},{}".format((i*batch_size)-batch_size+idl+1, len(image_dataset), inputs_paths[idl].split(in_dir)[1], class_names[img_preds[idl]], preds[idl], prob)
            file.write(print_str + '\n')
            print(print_str)
            shutil.copy(inputs_paths[idl], os.path.join(result_dir, class_names[img_preds[idl]], os.path.basename(inputs_paths[idl])))
        sys.stdout.flush()
        # statistics
        probs_history = probs_history + img_probs
        preds_history = preds_history + img_preds
        path_history = path_history + list(inputs_paths)
    
    time_elapsed = time.time() - since
    print('Demo complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    file.write('\nDemo complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))

    if dates == None:
        print('######### Results for {} ############'.format(in_dir))
        file.write('######### Results for {} ############\n'.format(in_dir))
    else:
        print('######### Results for {} {} ############'.format(word, dates))
        file.write('######### Results for {} {} ############\n'.format(word, dates))
    cum_results = []
    for idc, mask in enumerate(class_names):
        idp = idc if idc < 5 else -1
        cum_res = np.sum(np.array(preds_history)==idp)
        print_str = '  {}: {}'.format(mask, cum_res)
        print(print_str)
        file.write(print_str + '\n')
        cum_results.append(cum_res)
    print('######################################################\n')
    file.write('######################################################\n')
    file.write('{}\n'.format(class_names))
    file.write('{}\n'.format(cum_results))
    file.close()
    return cum_results, class_names


if __name__ == "__main__":

    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)
    weights_name = 'convnext-small_i224_b16_e50_imnet-10k'
    batch_size = 32
    full_results = []


    keywords = ["face_mask","n95","ffp2","cubrebocas","barbijo"]
    dates = "2019-12-01__2022-12-31"


    #################### Uncomment this section for demo classy only ##########################    
    dates = None
    ##################### Chose pahts for demo classy only ####################################
    data_dir = "./RetinaFace/res_Twitter"   
    out_dir = "./RetinaFace/res_Classy"
    ##################### Leave it as it is for Twitter mining ################################

    if dates == None:
        res, class_names = mask_classy(out_dir, data_dir, None, dates, weights_name, int(batch_size/2.0))
        print('\n{}\n{}'.format(class_names, res))
    else:
        for word in keywords:
            res, class_names = mask_classy(out_dir, data_dir, word, dates, weights_name, batch_size)
            full_results.append(res)
        results = np.sum(np.array(full_results),0)
        print('\n{}\n{}\n{}'.format(dates, class_names, results))

    