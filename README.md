# email-spam-classifier

This project is made to differentiate ham and spam email and putting them into two groups under multiple approaches.

> **Approach-1 : Naive-Bayes and bag of words**

- Loading the data
  - Corpus size = 5172 emails (enron1)

- Pre-Processing
  - Splitting the text by white spaces and punctuation marks using a tokenizer.
  - Linking the different forms of the same word (for example, “price” and “prices”, “is” and “are”) to each other – using a lemmatizer

- Extracting the features
  - Collected 5172 feature sets

- Training the classifier

  - Training set size = 4137 emails

  - Test set size = 1035 emails

- Evaluating the classifier.
 
  - Accuracy on the training set = 0.9605994682136814

  - Accuracy of the test set = 0.9304347826086956

  - Most Informative Features

               forwarded = 1                 ham : spam   =    148.2 : 1.0
                    2004 = 1                spam : ham    =    103.5 : 1.0
            prescription = 1                spam : ham    =    100.3 : 1.0
                     nom = 1                 ham : spam   =     91.2 : 1.0
                      xl = 2                 ham : spam   =     84.4 : 1.0
                    pain = 1                spam : ham    =     82.4 : 1.0
                     sex = 1                spam : ham    =     61.2 : 1.0
                featured = 1                spam : ham    =     57.9 : 1.0
                    spam = 1                spam : ham    =     56.3 : 1.0
                      cc = 2                 ham : spam   =     54.2 : 1.0
              nomination = 1                 ham : spam   =     51.7 : 1.0
            solicitation = 1                spam : ham    =     51.4 : 1.0
                thousand = 1                spam : ham    =     50.0 : 1.0
                creative = 1                spam : ham    =     49.8 : 1.0
              understood = 1                spam : ham    =     49.8 : 1.0
                deciding = 1                spam : ham    =     48.2 : 1.0
                      cc = 1                 ham : spam   =     47.8 : 1.0
                   cheap = 1                spam : ham    =     47.5 : 1.0
              compliance = 1                spam : ham    =     45.5 : 1.0
                 foresee = 1                spam : ham    =     44.9 : 1.0
                 
> **Approach-2 : CNN**

**in process...**
