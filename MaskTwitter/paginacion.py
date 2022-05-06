import tweepy
from MaskTwitter import credentials
import time
import os
import wget
client = tweepy.Client(bearer_token=credentials.BEARER_TOKEN)



def capturarTweets(palabra, date):
    i=0
    print('palabra:'+palabra+" date:",date)
    start_time = date[0]+"T00:00:00.000Z"
    end_time = date[1]+"T00:00:00.000Z"
    carpeta= palabra.replace(" ","_")
    query = palabra + ' -is:retweet has:media'
    numeroTotalImagenes = 0
    numeroTweets = 0  
    for response in tweepy.Paginator(client.search_all_tweets, query=query,
                                    start_time = start_time,
                                    end_time = end_time,
                                    media_fields=['preview_image_url','url'], expansions='attachments.media_keys',
                                    tweet_fields=['created_at'], max_results=100, limit=10000):
        numeroImagenes= 0
        numeroTweets += response.meta['result_count']
        i += 1
        print("Numero de query:"+str(i))
        time.sleep(1)
        media_files = []
        if __name__ == "__main__":
            os.makedirs('./'+carpeta, exist_ok=True)
            os.makedirs('./'+carpeta + '/' +date[0] + '__' + date[1], exist_ok=True)

        else:
            ruta = os.getcwd()
            ruta,ernest = os.path.split(ruta)
            ruta = ruta + '/MaskTwitter/'

            os.makedirs(ruta+carpeta, exist_ok=True)
            os.makedirs(ruta+carpeta + '/' +date[0] + '__' + date[1], exist_ok=True)
        
        if len(response.includes) ==0:
            continue
        media = {m["media_key"]: m for m in response.includes['media']}

        for tweet in response.data:
            tweet_data = tweet.data
            if not 'attachments' in tweet_data:
                continue
            attachments = tweet.data['attachments']
            media_keys = attachments['media_keys']
            if media[media_keys[0]].url:
                media_files.append(media[media_keys[0]].url)
        for media_file in media_files:
            try:
                if __name__ == "__main__":
                    ruta = './'+carpeta+ '/' +date[0] + '__' + date[1]

                else:
                    ruta = os.getcwd()
                    ruta,ernest = os.path.split(ruta)
                    ruta = ruta + '/MaskTwitter/'+carpeta+ '/' +date[0] + '__' + date[1]
                
                path, file = os.path.split(media_file)
                print("link: ",media_file)

                if not os.path.exists(ruta +'/' + file):
                    wget.download(media_file, ruta)
                    numeroImagenes += 1
            except Exception as e:
                print("Error descargar: ",e)
        numeroTotalImagenes += numeroImagenes
    if numeroImagenes == 0:
        time.sleep(3)

    print("numeroTweets:"+str(numeroTweets)+"numeroImagenes:"+str(numeroTotalImagenes))
    return(numeroTweets, numeroTotalImagenes)

if __name__ == "__main__":
    palabra = "face mask"
    date = ["2019-12-30","2019-12-31"]


    capturarTweets(palabra,date)



