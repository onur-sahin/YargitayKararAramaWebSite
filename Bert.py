
import datetime
from transformers import AutoModel, AutoTokenizer, optimization
from L2Retriever import L2Retriever
from sklearn.metrics.pairwise import euclidean_distances
import torch
import os
import pandas as pd
import sqlite3
import numpy as np
from tensorflow.keras.utils import Progbar


class Bert:


    modelNames =    [
                "dbmdz/bert-base-turkish-128k-cased"
                ]

    modelType = 0

    #device = torch.device("cuda")

    models = []

    tokenizers = []


    def __init__(self, modelNo=0):

        self.get_models()

        self.model = Bert.models[modelNo]

        self.tokenizer = Bert.tokenizers[modelNo]

        self.id_karar = []
        self.order_of_vector = []
        self.wordCount = []
        self.X_vects = []

        self.getData()



        self.X_vects = np.array(self.X_vects)


        self.X_square_norm = np.sum(self.X_vects**2, axis=1).reshape((-1,1))

        self.retriever = L2Retriever(768, use_norm=False, top_k=200, use_gpu=False)
    
        self.recommend = self.buildDocumentRecommender(self.id_karar, self.X_vects)




    def search_in_Bert(self, query_text):

        

      id_list, orderVector_list, wordCount_list  = self.recommend(query_text)

      results = {}

      if( list.__len__() > 0 ):

          cursor = self.connectDatabase()

          query = """
                  SELECT id, daire_ismi, esas_no, esas_yil,
                  karar_no, karar_yil, tarih, metin,
                  
                  FROM tbl_karar
  
                    INNER JOIN tbl_daire ON daire=id_daire
                    INNER JOIN tbl_mahkemesi ON mahkemesi=id_mahkeme

                  WHERE id in {}
                  """.format( tuple(id_list) )


          for row in cursor.execute(query):


            results[row[0]] = {
                              "id":         row[0],
                              "daire_ismi": row[1],
                              "esas_no":    str(row[2])+"/"+str(row[3]),
                              "karar_no":   str(row[4])+"/"+str(row[5]),
                              "tarih":      datetime.datetime.strptime(row[6], "%Y-%m-%d %H:%M:%S").strftime("%d/%m/%Y"),
                              "metin":      row[7],
                              "orderVector_list": row[8],
                              "wordCount_list" : row[9],
                              "orderVector_list":[],
                              "wordCount_list":[]
                              }


          for index, id in enumerate(id_list):
            results[id]["orderVector_list"].append(orderVector_list[index])
            results[id]["wordCount_list"].append(wordCount_list[index])

      return results

    def get_models(self):

        # device = torch.device("cuda")

        count = 0

        for i in Bert.modelNames:
            
            if(os.path.isdir("bertModels/"+i) == False):

                Bert.models.append( AutoModel.from_pretrained(i) )

            
                Bert.models[count].save_pretrained(i)

            else:

                Bert.models.append(  AutoModel.from_pretrained("bertModels/"+i) )

                

            if (os.path.isdir("bertTokenizers/" + i) == False):
                
                Bert.tokenizers.append( AutoTokenizer.from_pretrained(i) )
                Bert.tokenizers[count].save_pretrained("bertTokenizers/" + i )

            else:

                Bert.tokenizers.append(AutoTokenizer.from_pretrained("bertTokenizers/"+ i))


            count += 1

            # if( torch.cuda.is_available() ):
            #     i.to(device)


    def bert_vectorizer(self, text3, verbose=True):

        inputs= self.tokenizer(text3, return_tensors='pt', verbose=True, truncation=True, padding=True)

        # if( torch.cuda.is_available() ):
        #     inputs.to(device) #token gpu'ya aktarılır

        tnsr = self.model(**inputs)
        return tnsr[1]



    def connectDatabase(self):

        database = "yargitayKararlari_vektorlu.db"

        con = sqlite3.connect(database)

        return con.cursor()



    def getData(self):


        cursor = self.connectDatabase()


        for i in cursor.execute("SELECT COUNT(*) FROM tbl_karar"):
            count = i[0]
            break


        #count =2000

        bar = Progbar(count)


        count = 0
        
        query = """
                            SELECT id, vector, orderVector, word_count 
                            FROM tbl_karar
                            INNER JOIN tbl_vectors ON id=id_karar
        """

        for row in cursor.execute(query):
            
            count += 1

            self.id_karar.append(row[0])
            
            self.X_vects.append(  np.frombuffer( row[1], dtype=np.float32 ) )

            self.order_of_vector.append(row[2])

            self.wordCount.append(row[3])

            if ( count % 5000 == 0):
              bar.add(5000)

    

  
    def get_vectors(self, text):

        vectors = []

        paragraphs_list = self.slicer_by_paragraph(text)

        pure_paragraphs_list = self.purifyText(paragraphs_list)

        texts_sliced_by_512 = self.batch_list_sliced_by_512(pure_paragraphs_list, 512)

        for paragraph in texts_sliced_by_512:
            
            temp = self.bert_vectorizer(paragraph)

            vectors.append(   temp.detach().cpu().numpy()[0]   )

        return vectors





    def buildDocumentRecommender(self, id_karar, vectorized_plots, top_k=10):

      retriever = L2Retriever(vectorized_plots.shape[1], use_norm=True, top_k=top_k, use_gpu=False)

      vectorized_norm = np.sum(vectorized_plots**2, axis=1).reshape((-1,1))
      
      

      def recommend(query_text):

        vectors = self.get_vectors(query_text)

        list_kararId = [] #[karar_id]
        list_vectorOrder= []
        list_vector_wordCount = []

        try:
          idx = retriever.predict(vectorized_plots,
                                  vectors[0], 
                                  vectorized_norm)[0][1:]
          for i in idx:
            print (id_karar[i])

            list_kararId.append(  id_karar[i]  )

            list_vectorOrder.append (  self.order_of_vector[i]  )

            list_vector_wordCount( self.wordCount[i])


          return list_kararId, list_vectorOrder, list_vector_wordCount

        except ValueError:
          print("{} not found in movie db. Suggestions:")
          # for i, id in enumerate(self.karar_id):
            
          #   if query.lower() in id.lower():
          #     print(i, id)

              
      return recommend




    @staticmethod
    def purifyText(text_list):

      paragraphs_list = []

      for text in text_list:

        pureText = ""

        flag = True

        for j in text:

          if( j == "<"):
              flag = False

          if( j == ">"):
              flag = True
              continue

          if(flag == True):
              pureText += j

        pureText2= []

        for i in pureText.split(" "):
          if( i != ''):
            pureText2.append(i)

        
        if(pureText2.__len__() > 0):
        
            pureText2 = " ".join(pureText2)

            paragraphs_list.append(pureText2)


      return paragraphs_list


    @staticmethod
    def slicer_by_paragraph(text):

      list = []
      
      for i in text.split("<br>"):
        
        if( i != ''):
          list.append(i)

      return list




    def get_batch_by_size(self, paragraphs_list, size, first_paragraph=True):    #paragraphs_list bir liste 

      
      batch_size = size

      if( paragraphs_list.__len__() == 0):
        return ""
      
      words_list = paragraphs_list[0].split(" ")

      for index, word in enumerate(words_list):

        if(word == ''):
          words_list.pop(index)
      
      words_list_size = words_list.__len__()
      
      if (batch_size == 0):
        return ""

      min_batch_group_count = float(words_list_size)/float(batch_size) 

      if( min_batch_group_count <= 1 ):

        first = paragraphs_list.pop(0)

        tmp_batch = first  + " " + str( self.get_batch_by_size(   paragraphs_list, batch_size - words_list_size, first_paragraph=False   ) )
        
        return tmp_batch

      elif (min_batch_group_count > 1 and first_paragraph == True):

        if(words_list_size <= batch_size*1.1):
          return paragraphs_list.pop(0)

        else:


          
          index = batch_size-1

          while(index >= 0):

            if(words_list[index][-1] in [".", ":", "!", "?"]):
              break

            index -= 1

          if( index <= (batch_size-1)*0.9 ):
            index = batch_size
            
          
          word_head = paragraphs_list[0][0:index]

          word_tail = paragraphs_list[0][index:]

          if( word_tail.__len__() == 0):
            paragraphs_list.pop(0)

          else:
            paragraphs_list[0] = word_tail

          return word_head


      elif (min_batch_group_count > 1 and first_paragraph == False):

        # if(paragraphs_list.__len__() > 1):
        
        #   if( batch_size < paragraphs_list[1].split(" ").__len__() ):

        #     return ""

        index = batch_size-1

        while(index >= 0):

          if(words_list[index][-1] in [".", ":", "!", "?"] or words_list[index] in [".", ":", "!", "?"] ):
            break

          index -= 1

        if( index <= (batch_size-1)*0.9 ):
          index = batch_size-1

        while(index >= 0):

          if(paragraphs_list[0][index] != " " ):
            index -= 1

          else:
            break
        
        word_head = paragraphs_list[0][0:index]

        word_tail = paragraphs_list[0][index:]

        if( word_tail.__len__() == 0):
          paragraphs_list.pop(0)

        else:
          paragraphs_list[0] = word_tail

        return word_head




    def batch_list_sliced_by_512(self, paragraphs_list, size):

      paragraphs_list_sliced_by_size = []

      while (paragraphs_list.__len__() != 0):
        paragraphs_list_sliced_by_size.append( self.get_batch_by_size(paragraphs_list, size) )

      return paragraphs_list_sliced_by_size