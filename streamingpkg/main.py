from streamingpkg.youtubeprocessing import *


def main():
    print("Start of Main")

    SearchKeyword="Fintech"

    getyoutubecaptions(SearchKeyword)
    print("\nAccessing from MAIN\n")
    print("\nVideoMetaDataDF : \n",videoMetaDatadf)
    print("\nCaption DF: \n",Captiondf)


if __name__ == "__main__":
    main()

