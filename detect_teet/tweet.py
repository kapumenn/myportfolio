import tweepy
import csv


# 各種ツイッターのキーをセット
consumer_key = "nKdvsTw9SyjzFOXCRbm6HnHCT"
consumer_secret = "Z85jtUTTL4asYnmigvSLoFmLHkFfCGjsVNSFfMylruar3RwDZb"
access_key = "1467699883704713216-kSOOzZdc6PX4vqgQvfbpvOQapcKswX"
access_secret = "VhYic9ZjB4Xd3DEbNAbHTqUAOIP4hcT0Ln7oywvcJdlpK"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

#ツイート取得
tweet_data = []

for tweet in tweepy.Cursor(api.user_timeline,screen_name = "Suwa1129",exclude_replies = True).items():
    tweet_data.append([tweet.id,tweet.created_at,tweet.text.replace('\n',''),tweet.favorite_count,tweet.retweet_count])

#csv出力
with open('tweets.csv', 'w',newline='',encoding='utf-8') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["id","created_at","text","fav","RT"])
    writer.writerows(tweet_data)
pass