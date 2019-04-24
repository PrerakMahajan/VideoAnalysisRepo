from streamingpkg.youtubeprocessing import *
# from sklearn.svm import SVC



def main():



    print("Start of Main")


    SearchKeyword="Fintech Issues"

    getyoutubecaptions(SearchKeyword)
    print("\nAccessing from MAIN\n")
    print("\nVideoMetaDataDF : \n",videoMetaDatadf)
    videoMetaDatadf.to_csv("C:\\Users/sourabh.vermaLA/Desktop/Metdfff.csv")
    Captiondf.to_csv("C:\\Users/sourabh.vermaLA/Desktop/youtube/Capf.csv")
    print("\nCaption DF : \n",Captiondf)


if __name__ == "__main__":
    main()

