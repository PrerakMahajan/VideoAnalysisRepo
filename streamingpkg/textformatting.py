
import requests
import math
import re

'''
a = ",sdbncs ,sadmncmsd.ksjdk slkdc? ksjd "

for i in reversed(a):
    print(i)
    if (re.match('[.?!]',i)  is not None )is True:
        print("Broken on",i)
        break

'''

def getPunctuatedText(Caption):


    new_sentence=''
    sen_part = ''
    punctuated=''
    current_part=1
    end_var=0
    start_var=0

    if len(Caption) >= 53000 :
        parts = math.ceil(len(Caption) / 53000)
        print("________Caption for a particular Video has been split into ", parts,"parts. PLease wait while it gets Punctuated.")
        #print(parts,"PARTS")
        while current_part <= parts :                   #Parse all the parts
            start_var = end_var
            end_var = 53000*current_part
            if current_part == parts :                  #Parse till the End
                end_var = len(Caption)-1
            #print(current_part,start_var,end_var)
            if Caption[end_var] != " ":                #Check if the Sentence cut is ending on Space or Move back
                while Caption[end_var] != " ":
                    end_var = end_var - 1
                #print(end_var)
            sen_part = Caption[start_var:end_var]
            #print(sen_part)
            data = {
                'text': sen_part
            }

            response = requests.post('http://bark.phon.ioc.ee/punctuator', data=data)
            pu = response.text

            '''
            #TBD
            #Get to the last senetence
            #TBD--> Remove punctuations from last sentence and
            #punctuated = pu[0:max(pu.rfind("."),pu.rfind("?"),pu.rfind("!"))+1]
            #print("*******",punctuated)
            '''

            #print("**", pu)
            current_part = current_part +1
            new_sentence=new_sentence + pu
    else:
        data = {
            'text': Caption
        }

        response = requests.post('http://bark.phon.ioc.ee/punctuator', data=data)
        pu = response.text
        new_sentence = pu

    print(new_sentence)
    return(new_sentence)

#Caption="NULL"
#Caption = "{[Music] [Music] [Music] [Music] [Applause] [Music] [Applause] [Music] [Music] [Applause] [Music] [Applause] [Music] [Applause] [Music] [Applause] [Music] [Applause] [Music] [Applause] [Music] [Music] [Applause] [Music] [Applause] [Music] [Applause] [Music] [Applause] [Music] [Music] [Applause] [Music] [Music] [Applause] [Music] [Applause] [Music] [Applause] [Music] [Music] [Music] [Applause] [Music] [Applause] [Music] [Applause] [Music] [Applause] [Music] [Applause] [Music] [Applause] [Music] [Music] [Music] [Music] [Music] welcome to the FinTech face-off my name is Jennifer grizzle and I lead global marketing for services enterprise business here at LinkedIn the future of Finance is already here but clarity remains elusive the flare up and flame up of FinTech unicorns the race of incumbents to innovate while challengers scale the the shadow of big tech Demuth's and the portent of chinese platforms in many debates "
#getPunctuatedText(Caption)
