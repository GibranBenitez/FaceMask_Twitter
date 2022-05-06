import time
import os
import datetime
import busquedaTweets

while True:

	inicio = time.time()
	def date_tweet():
		dates_list=[]
		now = datetime.datetime.now()
		for number_x in range(2):
			seven=now-datetime.timedelta(days=number_x)
			dates_list.append(seven.strftime('%Y-%m-%d'))
		return dates_list[::-1]	
		


	keywords=["n95","face mask","cubrebocas","tapabocas","barbijo"]	
	dates = date_tweet()


	for keyword in (keywords):
		file = open("resultados_" + dates[0] + "--" + dates[-1] + ".txt", "a+")
		file.write(keyword+"\n")
		file.close()
		print(keyword)
		numeroImagenesxKeyword = 0
		numeroTweetsxKeyword = 0
		for ind_range in range (len(dates)):
			lastIDTweet = 0
			if ind_range + 1 != len(dates): 
				fechas = [dates[ind_range],dates[ind_range+1]]
				print(fechas)
				recorridoCompleto = False
				numeroFTweets = 0
				numeroFImagenes = 0
				while not(recorridoCompleto):
					estado, lastIDTweet, numeroTweets, numeroImagenes = busquedaTweets.capturarTweets(keyword,fechas,lastIDTweet)
					numeroFTweets += numeroTweets 
					numeroFImagenes += numeroImagenes
					if estado == 1:
						recorridoCompleto = True
					else:
						print('Espera 15 min')
						time.sleep(900)
				print(keyword + "  " + dates[ind_range]+"-" + dates[ind_range+1]+ " - " +str(numeroFTweets) + " - " + str(numeroFImagenes) + "\n")
				file = open("resultados_" + dates[0] + "--" + dates[-1] + ".txt", "a+")
				file.write(keyword + "  " + dates[ind_range]+"-" + dates[ind_range+1]+ " - " +str(numeroFTweets) + " - " + str(numeroFImagenes) + "\n")
				file.close()
				numeroImagenesxKeyword += numeroFImagenes
				numeroTweetsxKeyword +=	numeroFTweets
			else:
				print('Fin de fechas')
				print(keyword + "  " + dates[0]+"-" + dates[-1]+ " - " +str(numeroFTweets) + " - " + str(numeroFImagenes) + "\n")
				file = open("resultados_" + dates[0] + "--" + dates[-1] + ".txt", "a+")
				file.write(keyword + "  " + dates[0]+"-" + dates[-1]+ " - " +str(numeroTweetsxKeyword) + " - " + str(numeroImagenesxKeyword) + "\n")
				file.close()
			


	
	final = time.time()
	tiempo = round(final-inicio,0)
	esperar = (86400-tiempo) #86400 24hrs
	horas = time.strftime("%H:%M:%S", time.gmtime(esperar))
	print(f'Ya es toda, tiempo de ejecucion:{tiempo} segundos, esperar: {horas} hrs' )
	time.sleep(esperar)








