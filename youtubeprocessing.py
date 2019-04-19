from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
import pandas as pd
from oauth2client.tools import argparser


#GLOBAL Variables
videoMetaDatadf = pd.DataFrame
#VideoCaptionDF

def getyoutubecaptions(SearchKeyword):

    search_response = __setyoutubeconf(SearchKeyword,maxResults=4)
    searchedVideoslist = __Search(search_response)
    print(videoMetaDatadf)
    #print("________________VideoId and its Transcript")

    FinalList = []
    for eac in searchedVideoslist.keys():
        eac
        vid_data = YouTubeTranscriptApi.get_transcript(eac, languages=['en'])
        # print(len(vid_data))
        # print(YouTubeTranscriptApi.get_transcript(eac, languages=['en']))

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

            #print(type(searchedVideoslist[eac])) #tuple
            #print(searchedVideoslist[eac][0])
            key=eac
            videoKV[key] = fullTextString
            FinalList.append(videoKV)
    print(FinalList)
    Captiondf = pd.DataFrame(FinalList)
    #Captiondf = pd.DataFrame(FinalList,columns=['Key','Caption'])
    print(Captiondf)

            #print(videoKV)

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
    videos1={}
    for search_result in search_response.get("items", []):
        if search_result["id"]["kind"] == "youtube#video":
            #print(search_result["snippet"]["title"])
            #Videos is a KV Dictionary
            #search_response is a JSON

            key=search_result["id"]["videoId"]
            videos.setdefault(key,[])
            videos[key].append(search_result["snippet"]["title"])
            videos[key].append(search_result["snippet"]["publishedAt"])
            videos[key].append(search_result["snippet"]["description"])
            videos[key].append(search_result["snippet"]["channelTitle"])

    print(videos)
    print("Details of all the fetched Videos :",videos)
    global videoMetaDatadf
    videoMetaDatadf = pd.DataFrame.from_dict(videos, orient='index',columns=["title","publishedAt","description","channelTitle"])
    #print(videoMetaDatadf)
    return videos



