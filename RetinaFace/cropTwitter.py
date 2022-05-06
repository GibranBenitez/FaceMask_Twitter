from __future__ import print_function
import os
import glob
import sys
from os.path import basename
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
import pdb
import shutil

parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('--dataset_folder', default='../MaskTwitter/', type=str, help='dataset path')
parser.add_argument('--crop_folder', default='./res_Twitter/', type=str, help='Dir to save txt results')
parser.add_argument('--face_folder', default='./res_Twitter_face/', type=str, help='Dir to save txt results')
parser.add_argument('--img_ext', default='.jpg', type=str, help='extension de las imagenes')


parser.add_argument('--face_norm', default=0.2, type=float, help='espacio fuera del rostro')

parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')

parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def face_norm(x1, y1, x2, y2, ratio=0):
    h = y2-y1
    w = x2-x1
    if h>w:
        ny2, ny1 = y2+int(h*ratio), y1-int(h*ratio)
        res = h-w+1
        nx1 = x1 - int(res/2.0)-int(w*ratio)
        nx2 = x2 + int(res/2.0)+int(w*ratio)
    else:
        nx2, nx1 = x2+int(w*ratio), x1-int(w*ratio)
        res = w-h+1
        ny1 = y1 - int(res/2.0)-int(h*ratio)
        ny2 = y2 + int(res/2.0)+int(h*ratio)

    return nx1, ny1, nx2, ny2


def recortarImagenes(keywords,dates):
    dates = dates[0] + "__" + dates[1]
    for indice,keyword in enumerate(keywords):
        print(indice,keyword)
        keywords[indice]=keyword.replace(" ","_")
        print(keywords[indice])

    torch.set_grad_enabled(False)

    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    # print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    main_folder_list = os.listdir(args.dataset_folder)
    main_folder_list.sort()
    
    file = open("RetinaP--"+dates+".txt", "a+")
    file.write("------------------------------------------\n")
    file.close()


    for main_folder in main_folder_list:
        if not main_folder in keywords:
            continue
        
        print(main_folder)
        file = open("RetinaP--"+dates+".txt", "a+")
        file.write("****"+main_folder+"****"+"\n")
        file.close()
    


        main_folder_path = os.path.join(args.dataset_folder, main_folder)
        if os.path.isfile(main_folder_path):
            continue


        folder_list = os.listdir(main_folder_path)
        folder_list.sort()
        for folder in folder_list:
            if folder != dates or len(os.listdir(args.dataset_folder+"/"+main_folder+"/"+folder)) == 0:
                continue
            print('== {}: =='.format(folder))
            
            folder_path = glob.glob(os.path.join(main_folder_path, folder, "*"))
            #folder_path = glob.glob(os.path.join(main_folder_path, folder, "*{}".format(args.img_ext2)))
                


            folder_path.sort()
            num_images = len(folder_path)

            _t = {'forward_pass': Timer(), 'misc': Timer()}
            # pdb.set_trace()
            # testing begin
            for i, img_name in enumerate(folder_path):
                # image_path = testset_folder + img_name
                image_path = img_name
                img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
                img = np.float32(img_raw)

                resize = 1

                # Image Normalization for RetinaFace model
                im_height, im_width, _ = img.shape
                scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                img -= (104, 117, 123)      # pixel normalization
                img = img.transpose(2, 0, 1)
                img = torch.from_numpy(img).unsqueeze(0)
                img = img.to(device)
                scale = scale.to(device)

                # RetinaFace inference
                _t['forward_pass'].tic()
                loc, conf, landms = net(img)  # forward pass
                _t['forward_pass'].toc()
                _t['misc'].tic()
                priorbox = PriorBox(cfg, image_size=(im_height, im_width))
                priors = priorbox.forward()
                priors = priors.to(device)
                prior_data = priors.data
                boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
                boxes = boxes * scale / resize
                boxes = boxes.cpu().numpy()
                scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
                landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
                scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                       img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                       img.shape[3], img.shape[2]])
                scale1 = scale1.to(device)
                landms = landms * scale1 / resize
                landms = landms.cpu().numpy()

                # ignore low scores
                inds = np.where(scores > args.confidence_threshold)[0]
                boxes = boxes[inds]
                landms = landms[inds]
                scores = scores[inds]

                # keep top-K before NMS
                order = scores.argsort()[::-1]
                # order = scores.argsort()[::-1][:args.top_k]
                boxes = boxes[order]
                landms = landms[order]
                scores = scores[order]

                # do NMS
                dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                keep = py_cpu_nms(dets, args.nms_threshold)
                # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
                dets = dets[keep, :]
                landms = landms[keep]

                # keep top-K faster NMS
                # dets = dets[:args.keep_top_k, :]
                # landms = landms[:args.keep_top_k, :]

                dets = np.concatenate((dets, landms), axis=1)
                _t['misc'].toc()
                count = 0

                # ---- Saving detected faces ---------------------------------------------------
                file_name = os.path.basename(img_name)
                name, ext = os.path.splitext(file_name)
                save_name = os.path.join(args.face_folder, main_folder, folder, name + ".txt")
                dirname = os.path.dirname(save_name)
                if not os.path.isdir(dirname):
                    os.makedirs(dirname)
                crop_name = os.path.join(args.crop_folder, main_folder, folder, file_name)
                dirname = os.path.dirname(crop_name)
                if not os.path.isdir(dirname):
                    os.makedirs(dirname)
                with open(save_name, "w") as fd:
                    bboxs = dets
                    file_name = os.path.basename(save_name)[:-4] + "\n"
                    bboxs_num = str(len(bboxs)) + "\n"
                    fd.write(file_name)
                    fd.write(bboxs_num)
                    for box in bboxs:
                        if box[4] < args.vis_thres:
                            continue
                        b = list(map(int, box))
                        # crop                
                        x1, y1, x2, y2 = face_norm(b[0], b[1], b[2], b[3], args.face_norm)
                        crop_img = img_raw[y1:y2, x1:x2].copy()
                        
                        altura_imagen, ancho_imagen,canal = crop_img.shape #Capturando las dimensiones

                        try: 
                            
                            if(altura_imagen/ancho_imagen)<0.4 or (ancho_imagen/altura_imagen)< 0.4:
                                #print(f'TIRA:{altura_imagen} X {ancho_imagen}')
                                #print('image,path',image_path,'dirname: ',dirname)
                                #shutil.copy(image_path,dirname)
                                print("Imagen Original: ", file_name[:-1], 'Copiada' )
                                file = open("RetinaP--"+dates+".txt", "a+")
                                file.write(file_name)
                                file.close()




                            elif(altura_imagen<85) or (ancho_imagen<85):
                                print(f'Imagen omitida:{altura_imagen} X {ancho_imagen}')


                            else:
                                cv2.imwrite("{}_face{}{}".format(crop_name[:-4], count+1, args.img_ext), crop_img)

                            

                            
                                
                            
                            
                                
                                

                            
                        except:
                            print('     NO FACE')
                        x = int(box[0])
                        y = int(box[1])
                        w = int(box[2]) - int(box[0])
                        h = int(box[3]) - int(box[1])
                        confidence = str(box[4])
                        p1x, p1y = int(box[5]), int(box[6])
                        p2x, p2y = int(box[7]), int(box[8])
                        p3x, p3y = int(box[9]), int(box[10])
                        p4x, p4y = int(box[11]), int(box[12])
                        p5x, p5y = int(box[13]), int(box[14])
                        line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                        fd.write(line)
                        line = str(p1x) + " " + str(p1y) + " " + str(p2x) + " " + str(p2y) + " " + str(p3x) + " " + str(p3y) + " " + str(p4x) + " " + str(p4y) + " " + str(p5x) + " " + str(p5y) + " " + confidence + " \n"
                        fd.write(line)
                        count += 1

                print('   {:d}/{:d}: {:s} time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, name, _t['forward_pass'].average_time, _t['misc'].average_time))
                sys.stdout.flush()
                # pdb.set_trace()
                if count > 1:
                    print('     More than ONE FACE detected: {}'.format(count-1))







if __name__ == '__main__':
    keywords=["face mask","ffp2"]
    dates=["2020-03-11","2020-03-12"]

    recortarImagenes(keywords, dates)