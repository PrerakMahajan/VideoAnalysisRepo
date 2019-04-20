from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
import pandas as pd
from oauth2client.tools import argparser

from streamingpkg.textformatting import *


#GLOBAL Variables
videoMetaDatadf = pd.DataFrame(columns=['title', 'publishedAt', 'description', 'channelTitle'])
Captiondf = pd.DataFrame(columns=['Caption'])

def getyoutubecaptions(SearchKeyword):

    search_response = __setyoutubeconf(SearchKeyword,maxResults=4)
    __Search(search_response)       #Populates videoMetaDatadf

    #VideoId and its Transcript

    for eac in videoMetaDatadf.index.values:
        vid_data = YouTubeTranscriptApi.get_transcript(eac, languages=['en'])
        if not isinstance(vid_data, type(None)):
            # Parse the Array of Dictionary
            eachTextString = ''
            fullTextString = ''
            videoKV = {}
            i = 0
            while i < len(vid_data):
                eachTextString = ''.join(vid_data[i]['text'])
                fullTextString = fullTextString + " " + eachTextString
                i += 1
            # Add to Dataframe --> eac as 'Key' and to 'Caption' column data whose value is fullTextString


            global Captiondf
            Captiondf.loc[eac,'Caption']=getPunctuatedText(fullTextString)

    #print("___________CaptionDataFrame\n",Captiondf)
    return 0


def __setyoutubeconf(SearchKeyword,maxResults):


    # Declare Default Youtube Variables
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"
    DEVELOPER_KEY = "AIzaSyCxChOeTZQAXEa9an9rzo87_oecSisDOyc"

    # Building Youtube
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
    search_response = youtube.search().list(
        q=SearchKeyword,
        type="video",
        order="relevance",
        part="id,snippet",
        maxResults=maxResults,
        videoCaption="closedCaption",
        eventType="completed",
        publishedAfter=None,
        publishedBefore=None,
        topicId="Business,Technology"
    ).execute()
    #print("Search Response,",search_response)
    return search_response



def __Search(search_response):

    videos = {}

    for search_result in search_response.get("items", []):
        if search_result["id"]["kind"] == "youtube#video":
            #print(search_result["snippet"]["title"])
            #Videos is a KV Dictionary
            #search_response is a JSON

            key=search_result["id"]["videoId"]
            d = {'title': search_result["snippet"]["title"], \
                 'publishedAt': search_result["snippet"]["publishedAt"],\
                 'description':search_result["snippet"]["description"], \
                 'channelTitle':search_result["snippet"]["channelTitle"]
                 }
            global videoMetaDatadf
            videoMetaDatadf.at[key,:] = d

    #videoMetaDatadf = pd.DataFrame.from_dict(videos, orient='index')
    # print(videoMetaDatadf.index.values[0])
    #print("___________VideoMetaDataFrame\n",videoMetaDatadf)
    #return videoMetaDatadf



