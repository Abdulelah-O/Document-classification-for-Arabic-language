# Document-classification-for-Arabic-language

  ## Arabic text classification using the KALIMAT dataset, leveraging machine learning techniques for categorizing economy, religion, and sports articles.

  1.  **KALIMAT** Dataset:
     
       The **KALIMAT** dataset is one of the largest publicly available Arabic text corpora. It was created to assist in the development of Arabic NLP tools and has been widely adopted by researchers and developers alike. The dataset contains millions of Arabic words gathered from various sources, including online news platforms, blogs, and social media networks.

       **KALIMAT** is an Arabic natural language resource that consists of:

        1.	20,291 Arabic articles collected from the Omani newspaper Alwatan by (Abbas et al. 2011).
        2.	20,291 Extractive Single-document system summaries.
        3.	2,057 Extractive Multi-document system summaries.
        4.	20,291 Named Entity Recognised articles.
        5.	20,291 Part of Speech Tagged articles.
        6.	20,291 Morphologically Analyse articles.

        The data collection articles fall into six categories:
          **culture**, **economy**, **local-news**, **international-news**, **religion**, and **sports**.
          But in the code, I have used only **economy**, **religion** and **sports**.

  2. Preprocessing:

       	Apply preprocessing techniques specific to Arabic text, such as **tokenization**, **stemming**, and **normalization**.
       	Remove unnecessary elements such as **punctuation**, **special characters**, and **stopwords**.
       	create relevant features such as term frequency, TF-IDF vectors, or word embeddings.
     
  3. Model Selection and Training:

       	Select suitable machine learning or deep learning models (e.g., SVM, Naïve Bayes, KNN).
       	Train and validate the models on the preprocessed data, and tune hyperparameters to optimize performance.

  4. Evaluation:

       	Evaluate the model using appropriate metrics such as accuracy, precisdion, recall, and F1-score.


### How can you use the code?

  1. install the needed libraries for example: nltk and camel_tools etc.

  2. You need to download the dataset from the official website. https://sourceforge.net/projects/kalimat/

  3. Select the proper path for the dataset for example : "C:/Users/abode/NLP/NLP"
       
  4. Train the model

### Results:

  ![image](https://github.com/user-attachments/assets/2c6910f7-0a7b-4019-af7d-8a28ff3fa529)

  ![image](https://github.com/user-attachments/assets/61724090-6fd9-4ff7-b75b-8fab2c612dc3)

  ![image](https://github.com/user-attachments/assets/8ac1187b-cdaa-45aa-903b-be4b29aa9f69)





        
          

      

