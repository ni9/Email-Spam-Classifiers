# Email-Spam-Classifier

This project is made to differentiate ham and spam email and putting them into two groups under multiple approaches.

> ## Approach-1 : Naive-Bayes 

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

    

- **To Run the classifier**
    ```
    pip install -r requirements
    python Naive-Bayes\Naive-Bayes-Classifier.py
    ```    
#### Working Example
````    
E:\Pycharm\email-spam-classifier>python Naive-Bayes\Naive-Bayes-Classifier.py
Corpus size = 5172 emails
Collected 5172 feature sets
Training set size = 4137 emails
Test set size = 1035 emails
Accuracy on the training set = 0.9608411892675852
Accuracy of the test set = 0.9391304347826087
Most Informative Features
               forwarded = 1                 ham : spam   =    144.9 : 1.0
            prescription = 1                spam : ham    =     95.8 : 1.0
              compliance = 1                spam : ham    =     92.5 : 1.0
                    2001 = 1                 ham : spam   =     91.0 : 1.0
                    pain = 1                spam : ham    =     89.2 : 1.0
                     nom = 1                 ham : spam   =     88.2 : 1.0
                      xl = 2                 ham : spam   =     85.3 : 1.0
                    2005 = 1                spam : ham    =     79.2 : 1.0
                   cheap = 1                spam : ham    =     72.5 : 1.0
                   meter = 1                 ham : spam   =     69.3 : 1.0
                creative = 1                spam : ham    =     67.5 : 1.0
                featured = 1                spam : ham    =     64.2 : 1.0
                     ibm = 1                spam : ham    =     60.9 : 1.0
            solicitation = 1                spam : ham    =     57.5 : 1.0
                    2000 = 2                 ham : spam   =     56.2 : 1.0
                congress = 1                spam : ham    =     55.9 : 1.0
                deciding = 1                spam : ham    =     54.2 : 1.0
                    sony = 1                spam : ham    =     54.2 : 1.0
              understood = 1                spam : ham    =     54.2 : 1.0
                   cisco = 1                spam : ham    =     52.5 : 1.0

Do you want to test any email, press Y for yes and and N for no
y
Naive Bayes Spam Classifer

Enter the text which you want to classify

Enter email for testing as Spam/Ham. Press Ctrl+C to end the message
 > Subject: save your money by getting an oem software !
 > need in software for your pc ? just visit our site , we might have what you need . . .
 > best regards ,
 > joey
 >
The email you entered is a SPAM
---------------------------------------------------------------

Do you want to continue, press Y for yes and and N for no
y
Naive Bayes Spam Classifer

Enter the text which you want to classify

Enter email for testing as Spam/Ham. Press Ctrl+C to end the message
 > Subject: updated doorstep schedule
 > dear sally ,
 > here is the lastest . i ' ll continue to keep you in the loop .
 > best regards
 > shona
 >
The email you entered is a SPAM
---------------------------------------------------------------

Do you want to continue, press Y for yes and and N for no
y
Naive Bayes Spam Classifer

Enter the text which you want to classify

Enter email for testing as Spam/Ham. Press Ctrl+C to end the message
 > Subject: hpl status
 > the attached memo summarizes the status of the hpl transaction . please call me with any questions .
 > regards ,
 > brian
 >
The email you entered is a HAM
---------------------------------------------------------------

Do you want to continue, press Y for yes and and N for no
n
Session terminated
````
##### Note : Second example is reported as Spam but actually it is Ham


> ## Approach-2 : Neural Nets
- The current method used is:

    - Identify n ( default = 150 ) most frequent words in the corpus
    
    - Create a vector for each email in the corpus denoting whether 
    or not the n words appear in it. (0 if the word does not exist, 1 if the word exists)
    
    - Train a neural network on the vectorized data using stochastic gradient descent.

    - The training data and test data are 75% and 25% respectively of the loaded corpus. 
    
    - A classification accuracy of 97.04% is obtained on the test data using approximately 8424 emails.

    - The [NLTK](https://pypi.python.org/pypi/nltk) library is used to tokenize and create a frequency distribution of the words in the corpus. 
    
    - [Theano](https://pypi.python.org/pypi/Theano) is used to build a computational graph of the network and run it. 
    
    - The theano-specific code in this program is based on [Michael Nielsen's book](http://neuralnetworksanddeeplearning.com/) on Deep Learning.
    
    - **To Run the classifier**
        ```
        pip install -r requirements
        python Neural-Nets\Neural-Network-Classifier.py
        ```
    - An interactive session opens, where you can train and check the accuracy of the model, or enter custom text to test the model. 
    
    - Currently, the only way to modify the model(ie: add new layers, change the activation function, etc.) is to edit the source for this file.

    - The first run will take some time, but from the second time onwards enron's dataset(1-6) and most frequent words mfw( default n = 150 ) words will be cached, so it will run much quicker.
#### Working Example
````
E:\Pycharm\email-spam-classifier > python Neural-Nets\Neural-Network-Classifier.py

WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.

enron_ids:  [1, 2, 3, 4, 5, 6]

Considering top-150 words
Loading enron data
Loaded 33698 messages from enron data
Using 25274 messages as training data

Initialized Network:
NeuralNetwork(layers=[FullyConnectedLayer(150, 80, softmax), FullyConnectedLayer(80, 2, softmax), CrossEntropyCostLayer(2)])

Neural Nets Spam Classifer

Enter 1 for train and 2 for testing the classifier
1] Train
2] Test
 > 2

Enter text for testing as Spam/Ham. Press Ctrl+C to end the message
 > Subject: et & s photo contest - announcing the winners
 > congratulations to the following winners of the 2001 et & s photo contest .
 > over 200 entries were submitted ! the winning photos will be displayed in the
 > 2001 et & s public education calendar .
 > 


Found most frequent words cached. Unpickling...

Neural Net classifies it as: SPAM

Neural Nets Spam Classifer

Enter 1 for train and 2 for testing the classifier
1] Train
2] Test
 > 1

Enter epochs and eta separated by a space.
(Leave blank for epochs=60, eta=0.3)
 > 60 .3
Current epoch: 0
Accuracy on test data: 0.6622

Current epoch: 1
Accuracy on test data: 0.9023

Current epoch: 2
Accuracy on test data: 0.9249

Current epoch: 3
Accuracy on test data: 0.9323

Current epoch: 4
Accuracy on test data: 0.9341

Current epoch: 5
Accuracy on test data: 0.9365

Current epoch: 6
Accuracy on test data: 0.9385

Current epoch: 7
Accuracy on test data: 0.9382

Current epoch: 8
Accuracy on test data: 0.9387

Current epoch: 9
Accuracy on test data: 0.9404

Current epoch: 10
Accuracy on test data: 0.9418

Current epoch: 11
Accuracy on test data: 0.9429

Current epoch: 12
Accuracy on test data: 0.9432

Current epoch: 13
Accuracy on test data: 0.9450

Current epoch: 14
Accuracy on test data: 0.9462

Current epoch: 15
Accuracy on test data: 0.9463

Current epoch: 16
Accuracy on test data: 0.9476

Current epoch: 17
Accuracy on test data: 0.9483

Current epoch: 18
Accuracy on test data: 0.9495

Current epoch: 19
Accuracy on test data: 0.9498

Current epoch: 20
Accuracy on test data: 0.9501

Current epoch: 21
Accuracy on test data: 0.9500

Current epoch: 22
Accuracy on test data: 0.9505

Current epoch: 23
Accuracy on test data: 0.9515

Current epoch: 24
Accuracy on test data: 0.9527

Current epoch: 25
Accuracy on test data: 0.9533

Current epoch: 26
Accuracy on test data: 0.9540

Current epoch: 27
Accuracy on test data: 0.9549

Current epoch: 28
Accuracy on test data: 0.9555

Current epoch: 29
Accuracy on test data: 0.9565

Current epoch: 30
Accuracy on test data: 0.9563

Current epoch: 31
Accuracy on test data: 0.9569

Current epoch: 32
Accuracy on test data: 0.9569

Current epoch: 33
Accuracy on test data: 0.9581

Current epoch: 34
Accuracy on test data: 0.9582

Current epoch: 35
Accuracy on test data: 0.9589

Current epoch: 36
Accuracy on test data: 0.9595

Current epoch: 37
Accuracy on test data: 0.9601

Current epoch: 38
Accuracy on test data: 0.9607

Current epoch: 39
Accuracy on test data: 0.9613

Current epoch: 40
Accuracy on test data: 0.9613

Current epoch: 41
Accuracy on test data: 0.9620

Current epoch: 42
Accuracy on test data: 0.9625

Current epoch: 43
Accuracy on test data: 0.9628

Current epoch: 44
Accuracy on test data: 0.9628

Current epoch: 45
Accuracy on test data: 0.9631

Current epoch: 46
Accuracy on test data: 0.9635

Current epoch: 47
Accuracy on test data: 0.9645

Current epoch: 48
Accuracy on test data: 0.9659

Current epoch: 49
Accuracy on test data: 0.9664

Current epoch: 50
Accuracy on test data: 0.9664

Current epoch: 51
Accuracy on test data: 0.9667

Current epoch: 52
Accuracy on test data: 0.9673

Current epoch: 53
Accuracy on test data: 0.9681

Current epoch: 54
Accuracy on test data: 0.9686

Current epoch: 55
Accuracy on test data: 0.9689

Current epoch: 56
Accuracy on test data: 0.9695

Current epoch: 57
Accuracy on test data: 0.9701

Current epoch: 58
Accuracy on test data: 0.9704

Current epoch: 59
Accuracy on test data: 0.9704


Neural Nets Spam Classifer

Enter 1 for train and 2 for testing the classifier
1] Train
2] Test
 > 2

Enter text for testing as Spam/Ham. Press Ctrl+C to end the message

 > Subject: clyde drexler on espeak today !
 > join clyde drexler on espeak at ethink . enron . com , wednesday , june 14 at 10
 > a . m . houston time . clyde , an nba legend , will conduct an " open mike " session
 > to answer whatever questions you have for him .
 > are you at a remote location or can ' t make the event ? go into espeak now and
 > pre - submit your question ( s ) for clyde to answer during the scheduled event .
 > we want to answer everyone ' s questions , but due to the high volume of
 > questions we anticipate on this session , it would be helpful if you can keep
 > your questions short and simple . this will increase the opportunity for your
 > question to be answered .
 > ethink : invest your mind
 >



Neural Net classifies it as: HAM

Neural Nets Spam Classifer

Enter 1 for train and 2 for testing the classifier
1] Train
2] Test
 > 2

Enter text for testing as Spam/Ham. Press Ctrl+C to end the message
 > Subject: buy cheap viagra through us .
 > hi ,
 > we have a new offer for you . buy cheap viagra through our online store .
 > - private online ordering
 > - no prescription required
 > - world wide shipping
 > order your drugs offshore and save over 70 % !
 > click here : http : / / aamedical . net / meds /
 > best regards ,
 > donald cunfingham
 > no thanks : http : / / aamedical . net / rm . html
 >



Neural Net classifies it as: SPAM

Neural Nets Spam Classifer

Enter 1 for train and 2 for testing the classifier
1] Train
2] Test
 >
terminating...
Session terminated

````