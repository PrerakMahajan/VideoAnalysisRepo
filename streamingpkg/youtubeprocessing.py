from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
import pandas as pd
from oauth2client.tools import argparser
import iso8601
import pytz
from streamingpkg.textformatting import *
from nltk import tokenize


# GLOBAL Variables
videoMetaDatadf = pd.DataFrame(columns=['searchedKeyword','title', 'publishedAt', 'description', 'channelTitle'])
Captiondf = pd.DataFrame(columns=['Caption'])

####
Testdf= pd.DataFrame(columns=['Sentence'])
tempdf = pd.DataFrame(columns=['Sentence'])

videoMetaDatadf.index.names = ['VideoID']
Captiondf.index.names = ['VideoID']
####
tempdf1 = pd.DataFrame(columns=['videoId','Sentence'])

def getyoutubecaptions(SearchKeyword):

    search_response = __setyoutubeconf(SearchKeyword,maxResults=2)
    __Search(SearchKeyword,search_response)       # Populates videoMetaDatadf

    # VideoId and its Transcript

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

            punctext= getPunctuatedText(fullTextString)

             ################Testing DATAFRAME 2#############
            l = 0
            global tempdf1  ##For All videos

            while (l < len(Sentence_list)):
                # print(Sentence_list[l])
                tempdf1 = tempdf1.append({'videoId':eac,'Sentence':Sentence_list[l]},ignore_index=True)
                l=l+1
            print(tempdf1)

            #########ExtractTag#####

            getTags(tempdf1)
            ########################
            
            '''
            ################Testing DATAFRAME#############
            Sentence_list = tokenize.sent_tokenize(punctext)
            l=0

            while (l < len(Sentence_list)):
                global tempdf
                tempdf.loc[eac, 'Sentence'] = Sentence_list[l]
                global Testdf
                Testdf = Testdf.append(tempdf)
                l=l+1
            tempdf.drop(tempdf.index, inplace=True)
            print(Testdf)

            #############################################
            '''
            
            global Captiondf
            # Captiondf.loc[eac, 'Caption'] = fullTextString
            Captiondf.loc[eac,'Caption']=getPunctuatedText(punctext)

    # print(Captiondf[Captiondf.index.duplicated(keep=False)])
    # print(Captiondf.index.get_duplicates())
    # ids = Captiondf["title"]
    # Captiondf[ids.isin(ids[ids.duplicated()])].sort("title")
    # print("___________CaptionDataFrame\n",Captiondf)
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
        #order="viewCount",
        order="relevance",
        part="id,snippet",
        maxResults=maxResults,
        #videoCaption="closedCaption",
        #eventType="completed",
        publishedAfter=None,
        publishedBefore=None,
        #regionCode='IN'
        #topicId="Business,Technology"
    ).execute()
    #print("Search Response,",search_response)
    return search_response



def __Search(SearchKeyword,search_response):

    videos = {}

    for search_result in search_response.get("items", []):
        if search_result["id"]["kind"] == "youtube#video":
            #print(search_result["snippet"]["title"])
            #Videos is a KV Dictionary
            #search_response is a JSON
            _date_obj = iso8601.parse_date(search_result["snippet"]["publishedAt"])
            _date_utc = _date_obj.astimezone(pytz.utc)
            _date_utc_zformat = _date_utc.strftime('%d-%m-%Y')

            key=search_result["id"]["videoId"]
            d = {
                'searchedKeyword':SearchKeyword, \
                'title': search_result["snippet"]["title"], \
                 'publishedAt': _date_utc_zformat,\
                 'description':search_result["snippet"]["description"], \
                 'channelTitle':search_result["snippet"]["channelTitle"]
                 }
            global videoMetaDatadf
            videoMetaDatadf.at[key,:] = d

    #videoMetaDatadf = pd.DataFrame.from_dict(videos, orient='index')
    # print(videoMetaDatadf.index.values[0])
    #print("___________VideoMetaDataFrame\n",videoMetaDatadf)
    #return videoMetaDatadf



