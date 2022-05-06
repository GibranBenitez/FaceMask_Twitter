import cropTwitter
import purgeFalseTiras
import finalCount

def retinaface(keywords,dates):
	
	for keyword in keywords:

		cropTwitter.recortarImagenes(list([keyword]),dates)
	purgeFalseTiras.purgeFalse(keywords,dates)
	finalCount.confirmacion_imag(keywords,dates)





if __name__ == "__main__":
	keywords=["face mask"]
	dates=["2020-03-11","2020-03-12"]
	retinaface(keywords,dates)
	


#Diciembre 2019 -  dates=["2019-12-01","2019-12-31"]
#Enero 2020 - dates=["2020-01-01","2020-01-31"]
#Febrero 2020 -  dates=["2020-02-01","2020-02-29"]
#Marzo 2020 -  dates=["2020-03-01","2020-03-31"]
#Abril 2020 - dates=["2020-04-01","2020-04-30"]
#Mayo 2020 - dates=["2020-05-01","2020-05-31"]
#Junio 2020 - dates=["2020-06-01","2020-06-30"]
#Julio 2020 - dates=["2020-07-01","2020-07-31"]
#Agosto 2020 - dates=["2020-08-01","2020-08-31"]
#Septiembre 2020 - dates=["2020-09-01","2020-09-30"]
#Octubre 2020 - dates=["2020-10-01","2020-10-31"]
#Noviembre 2020 - dates=["2020-11-01","2020-11-30"]
#Diciembre 2020 - dates=["2020-12-01","2020-12-31"]



