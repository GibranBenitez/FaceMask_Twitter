import os



def confirmacion_imag(keywords,dates):
	file = open("resultadosFinales" +dates[0]+"__"+dates[1]+ ".txt", "a+")
	file.write("------------------------------------------\n")
	file.close()	

	
	
	for keyword in (keywords):
		keyword=keyword.replace(" ","_")
		file = open("resultadosFinales" +dates[0]+"__"+dates[1]+ ".txt", "a+")
		file.write(keyword+"\n")
		file.close()
		if not os.path.exists("./res_Twitter/"+keyword+"/"+dates[0]+"__"+dates[1]):
			continue
		numeroImagenes = len(os.listdir("./res_Twitter/"+keyword+"/"+dates[0]+"__"+dates[1]))
		file = open("resultadosFinales" +dates[0]+"__"+dates[1]+ ".txt", "a+")
		file.write(dates[0]+"__"+dates[1]+ " - "+ str(numeroImagenes) + "\n")
		file.close()
	


if __name__ == "__main__":
	keywords=["n95","ffp2","face mask","cubrebocas","barbijo"]
	dates= ["2019-12-30","2019-12-31"]

	confirmacion_imag(keywords,dates)			