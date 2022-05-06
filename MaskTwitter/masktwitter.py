import time
import os
import datetime
from MaskTwitter import paginacion





def masktwitter(keywords,dates):

	file = open("resultados_" + dates[0] + "--" + dates[1] + ".txt", "a+")
	file.write("------------------------------------------\n")
	file.close()
	
	for keyword in (keywords):
		file = open("resultados_" + dates[0] + "--" + dates[1] + ".txt", "a+")
		file.write(keyword+"\n")
		file.close()
		print(keyword+"---",dates)
		numeroTweets, numeroImagenes = paginacion.capturarTweets(keyword,dates)
		print(keyword + "  " + dates[0]+"-" + dates[1]+ " - " +str(numeroTweets) + " - " + str(numeroImagenes) + "\n")
		file = open("resultados_" + dates[0] + "--" + dates[1] + ".txt", "a+")
		file.write(dates[0]+"-" + dates[1]+ " - " +str(numeroTweets) + " - " + str(numeroImagenes) +"\n")
		file.close()
			



	print('Ya es toda')
	
	
if __name__ == "__main__":
	keywords=["face mask","n95","ffp2","cubrebocas","barbijo"]
	dates= ["2020-4-1","2020-4-15"]

	masktwitter(keywords,dates)
	






