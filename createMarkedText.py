
from types import new_class


def get_marked_text(dictionary, search_drive):


    wordCount = 0
    flag = True

    newMetin_list = []



    orderVector_list = dictionary["orderVector_list"]

    wordCount_list = dictionary["wordCount_list"]

    for word in dictionary["metin"].split(" "):
    
        if(word != ''):

            newMetin_list.append(word)


    newMetin2_list = []

    for order_vector in orderVector_list:

        markStart = 0
        markEnd = 0

        for i in range(0, order_vector):

            wordCount += wordCount_list[i]*i

            markStart = wordCount

        markEnd = wordCount + wordCount_list[order_vector]

        newMetin2_list.append("<mark>")
        newMetin2_list.append(newMetin_list[])

        


                