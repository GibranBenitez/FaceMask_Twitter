import tweepy
from tweepy import OAuthHandler
import datetime, time
import json
import wget
import credentials
import os

consumer_key = credentials.API_KEY
consumer_secret = credentials.API_SECRET_KEY
access_token = credentials.ACCESS_TOKEN
access_secret = credentials.ACCESS_TOKEN_SECRET

@classmethod
def parse(cls, api, raw):
    status = cls.first_parse(api, raw)
    setattr(status, 'json', json.dumps(raw))
    return status

tweepy.models.Status.first_parse = tweepy.models.Status.parse
tweepy.models.Status.parse = parse
tweepy.models.User.first_parse = tweepy.models.User.parse
tweepy.models.User.parse = parse

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)
#-------------------------------------------------------------




def capturarTweets(palabra, date, lastIDTweet):
    carpeta= palabra.replace(" ","_")
    print('palabra: ',palabra, "-- fechas: "+date[0] + "-" + date[1] + " ultimoTweet: "+ str(lastIDTweet))
    numeroImagenes = 0
    numeroTweets = 0
    tweets = 0
    while True:
        #Captura 200 tweets de una keyword especifica
        if lastIDTweet == 0:
            try:
                tweets = api.search_tweets(
                q=palabra,count=100,
                include_rts=False,
                exclude_replies=True,
                since=date[0],
                until=date[1])
                print("cantidad Tweets: ", len(tweets))
                if (len(tweets) == 0):
                    print("No se encontraron tweets de la keyword: ---", palabra, "--- durante estos días |", date[0] , " - ", date[1], "|")
                    return(1,lastIDTweet, numeroTweets, numeroImagenes)
                else:
                    numeroTweets += len(tweets)
                    lastIDTweet = tweets[-1].id-1
                    media_files = set()
            except Exception as e:
                print(e)
                print("Limite de peticiones 1")
                print("Ultimo tweet: " + str(lastIDTweet))
                print("Numero de Tweets: ", str(numeroTweets), "Numero Imagenes: ", str(numeroImagenes))
                return(0,lastIDTweet, numeroTweets, numeroImagenes)

        else:
            try:
                more_tweets = api.search_tweets(
                q=palabra,
                count=100,
                include_rts=False,
                exclude_replies=True,
                since=date[0],
                until=date[1],
                max_id=lastIDTweet-1)
                tweets = more_tweets
                print("cantidad Tweets: ", len(tweets))
                if (len(tweets) == 0):
                    print("No se encontraron tweets de la keyword: ---", palabra, "--- durante estos días |", date[0] , " - ", date[1], "|")
                    return(1,lastIDTweet, numeroTweets, numeroImagenes)
                else:
                    lastIDTweet = tweets[-1].id-1
                    numeroTweets += len(tweets)
                    media_files = set()
            except Exception as e:
                print(e)
                print("Limite de peticiones 2")
                print("Ultimo tweet: " + str(lastIDTweet))
                print("Numero de Tweets: ", str(numeroTweets), "Numero Imagenes: ", str(numeroImagenes))
                return(0,lastIDTweet, numeroTweets, numeroImagenes)


        os.makedirs('./'+carpeta, exist_ok=True)
        os.makedirs('./'+carpeta + '/' +date[0] + '__' + date[1], exist_ok=True)

        #Registra los tweets que contengan una imagen
        media = []
        for status in tweets:
            media = status.entities.get('media', [])
            if(len(media) > 0):
                print(media[0]['media_url'])
                media_files.add(media[0]['media_url'])
        #print("Media files: ",media_files)

        #Descarga las imagenes 
        for media_file in media_files:
            try:
                ruta = './'+carpeta+ '/' +date[0] + '__' + date[1]
                path, file = os.path.split(media_file)
                #print(file)
                if not os.path.exists(ruta +'/' + file):
                    wget.download(media_file, './'+carpeta+ '/' +date[0] + '__' + date[1])
                    numeroImagenes += 1
            except:
                print("Error descargar")

        media_files.clear()
        media.clear()

        if lastIDTweet == 0:
            lastIDTweet = tweets[-1].id
        else:
            # There are no more tweets
            if (len(tweets) == 0):
                print("Ultimo tweet: " + str(last_id))
                print('FIN')
                print("Numero de Tweets: ", str(numeroTweets), "Numero Imagenes: ", str(numeroImagenes))
                return(1,lastIDTweet, numeroTweets, numeroImagenes)
            else:
                lastIDTweet = tweets[-1].id-1


if __name__ == "__main__":
    palabra = 'cubrebocas'
    date = ('2022-02-24', '2022-03-02')
    lastIDTweet = 1496833119567196167

    capturarTweets(palabra,date, lastIDTweet)
