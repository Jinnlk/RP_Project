# Performance of Classifier Models on Detecting Fraudulent Job Postings Using TF-IDF Values
Jinnson Khen Lim
Student Association for Applied Statistics (SAAS), University of California, Berkeley jinnlk@berkeley.edu

## ABSTRACT

With the US’ slowing economy and the recent surge of tech industry layoffs, many job-seekers will be rushing to online hiring platforms such as Indeed, LinkedIn, and Glassdoor in hopes of securing a working position. However, this poses a major security concern since many listings require applicants to provide sensitive personal information. Especially when applying to hundreds of roles, it would be ideal if there was a method to identify fraudulent job postings to protect personal privacy and ultimately reduce wasted time. The goal of this study attempts to solve this dilemma by evaluating the effectiveness of different classification models on detecting fraudulent job postings using TF-IDF values extracted from the job descriptions. The Employment Scam Aegean Dataset (EMSCAD) which contains 17,880 real-life job postings, was used for this study. We trained five common classification models with their performance measured by running a classification report. Each of the models scored highly on all metrics; an average score of ninety on accuracy and f-1. 

### KEYWORDS: natural language processing (NLP), TF-IDF, classification, regex, scikit-learn

## INTRODUCTION

“Some of the world’s biggest tech companies have collectively laid off more than 150,000 workers in recent months.” (Forbes, 10 January 2023) Entering 2023, headlines like this are not uncommon to see in the news. Recently, giant tech corporations like Meta, Amazon, and Salesforce have decided to layoff thousands of employees. Some speculate this as a consequence from the rapid expansion of the tech sector after the recovery period of the pandemic. After years of over recruiting, now firms face the repercussions and must make harsh decisions to cut their workforce. 
With these recent surges in tech industry lay-offs, there will be an increase of individuals seeking job opportunities through online hiring platforms such as LinkedIn, Indeed, and Glassdoor to name a few. These websites are amazing for recruiters and applicants alike, providing wide exposure for various job openings, quick streamlines to apply to hundreds of positions, and an easy process for employers to post job listings. However, some may take advantage of these simplified methods with harmful intentions through fraudulent job postings. Especially for applicants, these fake listings can waste valuable time during the recruiting process, but in some serious cases, individuals can be victims of identity theft. As many postings require applicants to provide sensitive personal information, it is not surprising that corrupt persons can abuse provided data. Therefore, it is crucial for applicants to be confident that the position that they are applying for is authentic and ensures personal security. 

In this study, we will be exploring ways to combat these fraudulent job postings through a machine learning approach. Specifically, training and evaluating the performance of common classification models trained on TF-IDF values, a mathematical statistic used in NLP techniques. By the end of this paper, we hope to determine if this approach will be effective for pointing out fraudulent job postings. 

## METHODOLOGY

The dataset selected to conduct this study is the Employment Scam Aegean Dataset (EMSCAD) which contains 17,880 real-life job postings separated between 17,014 real and 866 fraudulent listings. The data was collected from various job advertisements from 2012-2014 by The University of the Aegean. 

The csv file was uploaded onto a python notebook in the VSCode editor using the Pandas library for data manipulation and NumPy library for array operations. Following through with the process of exploratory data analysis, I began to note important observations.  There are 17 feature columns, although only the class indicator column and qualitative columns will be used. One of the main concerns with the dataset is the class imbalance between real and fraudulent entries. In addition, the columns that will be utilized include various html tags, hyperlinks, and carriage returns that must be cleaned out.  

The first step of the process was to construct a clean and organized data frame to be used as the corpus for extracting TF-IDF values. All of the feature columns that contained descriptions from the original job posting were selected to represent the corpus. These columns were “description”, “requirements”, “benefits”, “company_profile”. For simplicity, the columns were merged into one column and relabeled as “description”.


![Image](https://github.com/users/Jinnlk/projects/1/assets/115912358/0537f14f-0da1-4b33-ab95-9060e6b9307a)


To clean the description column, three functions were defined to identify and remove all the different non-standard text within the descriptions. To accomplish this I imported ReGex, which is a module that uses standard expressions to identify and isolate certain patterns within strings. Within the functions, standard expressions were used to find non-standard words and replace them with empty strings, inherently removing them from the job descriptions. After applying each function on the column, the result was a clean and usable data frame of the different job listings within the dataset.


![Image](https://github.com/users/Jinnlk/projects/1/assets/115912358/1f36d685-403c-4fab-9849-c98110f66a65)


One small modification towards the final dataset was one hot encoding the ‘class’ column from a categorical vector to a binary vector. Since the column only included ‘f’ and ‘t’ strings, it was relatively simple to replace them with 1 and 0 values respectively. 

Although one of the main fundamental issues with the data is the major class imbalance. There are over 17,000 entries that are real job postings while there are only 800 that are fraudulent. If different models were to train on this imbalanced dataset, the results will be overtrained towards the majority class (“real”) and underperform when attempting to identify the minority class (“fraudulent”). For this case, I decided to downsample the majority class. The main reasoning behind this was that the quantity difference between the two classes was too large. Upsampling 800 entries to over 17000 will most likely result in overfitting since the entries will have to almost be replicated twenty times. On the other hand, downsampling the majority class will be a better approach to dealing with the imbalance. Reducing the number of real entries will still lead to a dataset of 1700 total job entries which is plenty considering the methods in this study. 

Using Pandas, I isolated the ‘real’ entries in the dataset (‘fraudulent’ == 0) and randomly selected 866 entries using the .sample function and appending this dataframe to the “fraudulent” entries data frame to have a final table that is cleaned, balanced, and now ready to extract TF-IDF values from. 


![Image](https://github.com/users/Jinnlk/projects/1/assets/115912358/65021435-761e-477c-b258-f712ca326225)


Term Frequency - Inverse Document Frequency is a mathematical statistic that uses the frequency of a term within a corpus to reflect its relevance within the different documents. It can be broken down into two different components. Term Frequency, which is the number of times a word appears within a document divided by the number or words within said document and Inverse Document Frequency which is the number of documents divided by the number of documents that contain the term. 


![Image](https://github.com/users/Jinnlk/projects/1/assets/115912358/a3d13fc7-6fe0-4892-b118-06848934a05d)


The intuition behind this technique is that TF-IDF values can give specific words higher importance compared to others within a document which can be used to uniquely characterize documents. For example, in the context of this study, an entry that has high TF-IDF values for ‘python’, ‘C’, and ‘SQL’ will most likely relate towards a Software Engineering job listing. Although, an entry that has high values for ‘creativity’, ‘Adobe’, or ‘Canva’ will most likely pertain towards a job in Graphic Design. 

In order to get the tf-idf values for all the unique terms within the corpus and represent them for each job description, I will be using sklearn’s Tfidf_Vectorizer. It is a pipeline that is equivalent to using Count_Vectorizer followed by Tfidf_Transformer. The Count_Vectorizer is a function that takes in a collection of text documents and converts them into a sparse matrix of counts for each unique term for each document. The Tfidf_Vectorizer function takes in a term count matrix and computes the TF-IDF values for each item. So, when applying the Count_Vectorizer function on the ‘corpus’ column the result is a sparse matrix of tf-idf values for all of the unique terms within the corpus for every job description.


![Image](https://github.com/users/Jinnlk/projects/1/assets/115912358/3855af4f-d3b4-47cb-9f08-d436b70a6bd8)


Now having all of the data pre-processed, we can now train models. The matrix of tf-idf values will be, ‘X’, the data, while the ‘fraudulent’ column will be ‘y’, the true class of each job listing.

Five commonly used classification models were selected to be trained on the data including, Random Forest (RF), Logistic Regression (LR), Support Vector Classification(SVC), Naive Bayes(NB), and K Nearest Neighbors (KNN). In order to measure the performance of each model, the data was split into a training and testing set using sklearn’s test_train_split function with a test_size of .30. The metrics selected to evaluate the predictions of each model will be accuracy, precision, recall, f-1, and support, using skelarn’s builtin metric function, classification_report. To account for overfitting, each of the models were run through a K-Fold cross validation test using sklearn’s cross_val_score function. The process runs multiple test-train splits on different “folds” or splits of the data to account for instances where the model may be trained to overfit on specific splits of data. Considering the size of the data, five k folds (k = 5) were selected as the parameters for the function. 
	
## RESULTS

After fitting all the models and evaluating their respective predictions, the table below visualized the performance of each model on each of the selected metrics.


![Image](https://github.com/users/Jinnlk/projects/1/assets/115912358/b3ef180b-bab9-4ebb-b923-3d5530973b5e)


The table below shows the K-Fold scores for each specific model. 


![Image](https://github.com/users/Jinnlk/projects/1/assets/115912358/c07409f3-b2b2-4a42-9f8f-6e310dc63607)


## CONCLUSION

Overall, considering the performance results above, it is safe to conclude that using TF-IDF values is an effective NLP technique to train classification models. The results were surprising as all of the models consistently scored high on all metrics with an average score within the 90’s range. For the K-Fold cross validation scores, each model scored high as well, averaging within the mid 80’s. The highest performing model was Support Vector Classification with Random Forest following just behind. One of the reasons SVC was able to perform so well, was the advantage it has with higher dimensional data. Since the tf-idf matrix contained over 17,000 features, it was apparent that SVC would have an upper hand. 

## ACKNOWLEDGEMENTS

I’d like to personally thank all my directors for their collaborative effort assisting me throughout this whole process. From narrowing down the research topic with Arnav G., to proposing the idea to use TF-IDF values with Clarie W., and of course Seanne C. for the amazing energy during our weekly workshops, I really feel like I was able to grow academically and professionally in the best SAAS committee, RP.  

## REFERENCES
Casselman, Ben, and Lauren Leatherby. “How Is the Economy Doing?” The New York Times, The New York Times, 13 Sept. 2022, https://www.nytimes.com/interactive/2022/09/13/business/economy/us-economy.html.
De Witte, Melissa. “What Explains Recent Tech Layoffs, and Why Should We Be Worried?” Stanford News, 2 Dec. 2022, https://news.stanford.edu/2022/12/05/explains-recent-tech-layoffs-worried/.
Madan, Rohit. “TF-IDF/Term Frequency Technique: Easiest Explanation for Text Classification in NLP with Python.” Medium, Analytics Vidhya, 27 Nov. 2019, https://medium.com/analytics-vidhya/tf-idf-term-frequency-technique-easiest-explanation-for-text-classification-in-nlp-with-code-8ca3912e58c3.
Marr, Bernard. “The Real Reasons for Big Tech Layoffs at Google, Microsoft, Meta, and Amazon.” Forbes, Forbes Magazine, 1 Feb. 2023, https://www.forbes.com/sites/bernardmarr/2023/01/30/the-real-reasons-for-big-tech-layoffs-at-google-microsoft-meta-and-amazon/?sh=373169692b67.
