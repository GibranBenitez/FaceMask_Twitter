import sys
sys.path.append("..")
from MaskTwitter import masktwitter
import RetinaFace



def facemaskTwiter(keywords,dates):

	masktwitter.masktwitter(keywords,dates)
	RetinaFace.retinaface(keywords,dates)




if __name__ == '__main__':
	keywords=["n95","ffp2","face mask","cubrebocas","barbijo"]
	dates=["2022-04-01","2022-04-02"]

	facemaskTwiter(keywords,dates)


