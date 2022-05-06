import os
from os import remove
import numpy as np
import cv2
import re




def purgeFalse(keywords,dates):
    dirnames = os.listdir('./res_Twitter')
    dates = dates[0]+"__"+dates[1]
    #print(dirnames)
    for directorio in dirnames:
        if not directorio in keywords:
            continue
        filenames = os.listdir("./res_Twitter/"+directorio)
        for filename in filenames:
            if filename !=dates:
                continue
            imageList = os.listdir("./res_Twitter/"+directorio+ "/"+ filename)
            for image in imageList:
                if image.find("_face") == -1:
                    #print(image)
                    for comparation in imageList:
                        ruta, extension = os.path.splitext(image)
                        comparar = ruta + "_face"
                        
                        
                        #print("ELIMINAR IMAGEN: " + comparar +"---" + comparation)
                        if comparation.find(comparar) != -1:
                            print("ELIMINAR IMAGEN: " + comparar +"---" + comparation)
                            remove("./res_Twitter/"+directorio+ "/"+ filename+"/"+image)
                            break



        


if __name__ == '__main__':
    keywords=["n95","ffp2","cubrebocas","barbijo"]
    dates= ["2019-12-30","2019-12-31"]

    purgeFalse(keywords,dates)



