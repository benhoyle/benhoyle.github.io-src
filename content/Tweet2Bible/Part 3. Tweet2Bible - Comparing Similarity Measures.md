
Title: 3. Tweet2Bible - Comparing Similarity Measures
Tags: initial_model
Authors: Ben Hoyle
Summary: This post looks at some approaches for matching tweets to Bible passages.

# 3. Tweet2Bible - Comparing Similarity Measures

Now we have our data we can look at some matching.

To start we will look at a number of off-the-shelf similarity functions. We will then compare these subjectively and see what gets us the best matches.

Note the docker-machine virtual machine only has one CPU and 1GB RAM - we need to run docker without the VM...

## Similarity Functions

Here are some initial similarity functions we can look at:

* [Difflib's SequenceMatcher](https://docs.python.org/3/library/difflib.html) has a "ratio" function that provides a match score for two strings. This represents a "naive" baseline.
* We can use spaCy's ["similarity" method](https://spacy.io/usage/vectors-similarity) on "doc" objects (i.e. as applied to each string).
* We can apply the techniques set available in Gensim as set out in [this helpful tutorial](https://radimrehurek.com/gensim/tut3.html).

We can then use the results as a baseline for more complex models and algorithms.

We will also time how long each method takes.

### Load Data


```python
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
```


```python
import pickle
with open("processed_data.pkl", 'rb') as f:
    tweets, bible_data = pickle.load(f)
```


```python
print("We have {0} tweets.".format(len(tweets)))
print("We have {0} Bible passages.".format(len(bible_data)))
```

    We have 9806 tweets.
    We have 31102 Bible passages.


### Difflib SequenceMatcher


```python
from difflib import SequenceMatcher

def similar(a, b):
    """Get a similarity metric for strings a and b"""
    return SequenceMatcher(None, a, b).ratio()

def get_matches(tweet, bible_data):
    """Match a tweet against the bible_data."""
    # Get matches
    scores = [
        (verse, passage, similar(tweet, passage)) 
        for verse, passage in bible_data
    ]
    # Sort by descending score
    scores.sort(key=lambda tup: tup[2], reverse = True) 
    return scores

def test_random_tweets(tweets, bible_data, n=5, k=5):
    """Print n examples for k tweets selected at random."""
    import random
    num_tweets = len(tweets)
    indices = random.sample(range(0, num_tweets), k)
    for i in indices:
        tweet = tweets[i]
        print("-----------------")
        print("Tweet text: {}".format(tweet))
        scores = get_matches(tweet, bible_data)
        for verse, passage, score in scores[0:n]:
            print("\n{0}, {1}, {2}".format(verse, passage, score))
```


```python
test_random_tweets(tweets, bible_data)
```

    -----------------
    Tweet text: "In addition, along with the advance of the electronic information society, a variety of electronic devices are utilized." #thetimeswelivein
    
    Mark 8:19, When I broke the five loaves among the five thousand, how many baskets full of broken pieces did you take up? They told him, Twelve., 0.4338235294117647
    
    Exodus 6:16, These are the names of the sons of Levi according to their generations: Gershon, and Kohath, and Merari; and the years of the life of Levi were one hundred thirty-seven years., 0.43174603174603177
    
    Numbers 3:21, Of Gershon was the family of the Libnites, and the family of the Shimeites: these are the families of the Gershonites., 0.4186046511627907
    
    Job 4:10, The roaring of the lion, and the voice of the fierce lion, the teeth of the young lions, are broken., 0.4166666666666667
    
    2 Corinthians 3:9, For if the service of condemnation has glory, the service of righteousness exceeds much more in glory., 0.4132231404958678
    -----------------
    Tweet text: RT @Dr_Cuspy: Why Watson and Siri Are Not Real AI http://t.co/s5MsxRK7Nd via @PopMech [Hofstadter pops up again; a renaissance?]
    
    Exodus 2:22, She bore a son, and he named him Gershom, for he said, I have lived as a foreigner in a foreign land., 0.35807860262008734
    
    2 Kings 3:22, They rose up early in the morning, and the sun shone on the water, and the Moabites saw the water over against them as red as blood., 0.35384615384615387
    
    Job 30:20, I cry to you, and you do not answer me. I stand up, and you gaze at me., 0.35175879396984927
    
    Job 38:26, To cause it to rain on a land where no man is; on the wilderness, in which there is no man;, 0.3470319634703196
    
    Genesis 34:31, They said, Should he deal with our sister as with a prostitute?, 0.34554973821989526
    -----------------
    Tweet text: EPO - The Administrative Council has been busy: updates concerning international supplementary searches, fees & search sharing coming up...
    
    Judges 1:21, The children of Benjamin did not drive out the Jebusites who inhabited Jerusalem; but the Jebusites dwell with the children of Benjamin in Jerusalem to this day., 0.3933333333333333
    
    2 Timothy 2:18, men who have erred concerning the truth, saying that the resurrection is already past, and overthrowing the faith of some., 0.39080459770114945
    
    Acts 15:9, He made no distinction between us and them, cleansing their hearts by faith., 0.39069767441860465
    
    Hebrews 11:38, (of whom the world was not worthy), wandering in deserts, mountains, caves, and the holes of the earth., 0.3884297520661157
    
    Joshua 13:28, This is the inheritance of the children of Gad according to their families, the cities and its villages., 0.3868312757201646
    -----------------
    Tweet text: Historians (and journalists) are always going to be important. https://t.co/o4s7DJEYwE
    
    Leviticus 9:16, He presented the burnt offering, and offered it according to the ordinance., 0.40993788819875776
    
    Job 36:33, Its noise tells about him, and the livestock also concerning the storm that comes up., 0.4093567251461988
    
    Proverbs 16:11, Honest balances and scales are Yahweh's; all the weights in the bag are his work., 0.40718562874251496
    
    John 20:10, So the disciples went away again to their own homes., 0.4057971014492754
    
    Acts 3:1, Peter and John were going up into the temple at the hour of prayer, the ninth hour., 0.40236686390532544
    -----------------
    Tweet text: Tip: use Chromium on Karmic EeePC in full screen (F11): excellent use of limited space.
    
    Isaiah 28:29, This also comes forth from Yahweh of Armies, who is wonderful in counsel, and excellent in wisdom., 0.44324324324324327
    
    Psalm 139:15, My frame wasn't hidden from you, when I was made in secret, woven together in the depths of the earth., 0.4126984126984127
    
    Hebrews 1:4, having become so much better than the angels, as he has inherited a more excellent name than they have., 0.4105263157894737
    
    Job 12:21, He pours contempt on princes, and loosens the belt of the strong., 0.40789473684210525
    
    Deuteronomy 25:10, His name shall be called in Israel, The house of him who has his shoe untied., 0.4024390243902439


### spaCy String Similarity

The 'en_core_web_lg' file crashed my Jupyter kernel but the 'en_core_web_sm' file loaded okay. I'll try the medium-sized file 'en_core_web_md'. Yes - 'md' file loaded okay.

Spacy uses an [average of the word vectors in a span or doc](https://spacy.io/usage/vectors-similarity#custom-similarity). (It may be faster to join the passages as a doc - process then split into spans.)


```python
!python3 -m spacy download en_core_web_md
```

    Collecting https://github.com/explosion/spacy-models/releases/download/en_core_web_md-2.0.0/en_core_web_md-2.0.0.tar.gz
    [?25l  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_md-2.0.0/en_core_web_md-2.0.0.tar.gz (120.8MB)
    [K    100% |################################| 120.9MB 7.0MB/s ta 0:00:011 0% |                                | 1.2MB 2.2MB/s eta 0:00:56    2% |                                | 2.8MB 4.0MB/s eta 0:00:30    2% |                                | 3.3MB 3.9MB/s eta 0:00:31    3% |#                               | 4.1MB 5.6MB/s eta 0:00:21    3% |#                               | 4.4MB 3.7MB/s eta 0:00:32    5% |#                               | 6.8MB 7.2MB/s eta 0:00:16    8% |##                              | 9.8MB 6.0MB/s eta 0:00:19    8% |##                              | 10.8MB 6.1MB/s eta 0:00:19    9% |##                              | 11.2MB 6.3MB/s eta 0:00:18    9% |###                             | 11.5MB 5.1MB/s eta 0:00:22    12% |###                             | 14.7MB 4.9MB/s eta 0:00:22    12% |###                             | 15.0MB 4.9MB/s eta 0:00:22    12% |####                            | 15.3MB 6.0MB/s eta 0:00:18    13% |####                            | 16.0MB 5.2MB/s eta 0:00:21    13% |####                            | 16.4MB 5.6MB/s eta 0:00:19    15% |####                            | 18.1MB 7.9MB/s eta 0:00:13    16% |#####                           | 19.6MB 7.2MB/s eta 0:00:15    16% |#####                           | 20.0MB 6.4MB/s eta 0:00:16    17% |#####                           | 21.4MB 5.7MB/s eta 0:00:18    19% |######                          | 23.4MB 8.2MB/s eta 0:00:12    19% |######                          | 24.0MB 6.3MB/s eta 0:00:16    20% |######                          | 24.4MB 9.9MB/s eta 0:00:10    20% |######                          | 25.0MB 5.3MB/s eta 0:00:19    21% |######                          | 25.7MB 3.7MB/s eta 0:00:26    22% |#######                         | 26.7MB 4.6MB/s eta 0:00:21    25% |########                        | 30.3MB 5.2MB/s eta 0:00:18    26% |########                        | 31.6MB 6.3MB/s eta 0:00:15    27% |########                        | 33.6MB 4.5MB/s eta 0:00:20    28% |#########                       | 35.0MB 8.4MB/s eta 0:00:11    29% |#########                       | 36.0MB 8.7MB/s eta 0:00:10    30% |#########                       | 37.4MB 7.6MB/s eta 0:00:12    33% |##########                      | 40.2MB 7.2MB/s eta 0:00:12    33% |##########                      | 40.7MB 8.6MB/s eta 0:00:10    35% |###########                     | 43.1MB 4.7MB/s eta 0:00:17    39% |############                    | 47.7MB 5.6MB/s eta 0:00:14    46% |##############                  | 55.8MB 9.0MB/s eta 0:00:08    46% |##############                  | 56.2MB 7.3MB/s eta 0:00:09    47% |###############                 | 57.5MB 4.9MB/s eta 0:00:13    50% |################                | 61.4MB 4.5MB/s eta 0:00:14    52% |################                | 63.4MB 7.0MB/s eta 0:00:09    52% |################                | 63.7MB 7.7MB/s eta 0:00:08    53% |#################               | 64.3MB 8.3MB/s eta 0:00:07    54% |#################               | 65.3MB 6.3MB/s eta 0:00:09    54% |#################               | 65.6MB 5.4MB/s eta 0:00:11    55% |#################               | 67.3MB 5.8MB/s eta 0:00:10    56% |##################              | 68.4MB 6.2MB/s eta 0:00:09    56% |##################              | 68.7MB 6.0MB/s eta 0:00:09    57% |##################              | 69.0MB 5.2MB/s eta 0:00:10    57% |##################              | 69.5MB 8.3MB/s eta 0:00:07    57% |##################              | 69.9MB 7.6MB/s eta 0:00:07    59% |###################             | 72.0MB 7.6MB/s eta 0:00:07    59% |###################             | 72.3MB 4.7MB/s eta 0:00:11    61% |###################             | 74.3MB 4.9MB/s eta 0:00:10    62% |###################             | 75.0MB 5.7MB/s eta 0:00:09    62% |###################             | 75.3MB 5.6MB/s eta 0:00:09    63% |####################            | 76.3MB 6.1MB/s eta 0:00:08    64% |####################            | 77.4MB 5.6MB/s eta 0:00:08    67% |#####################           | 82.2MB 4.0MB/s eta 0:00:10    70% |######################          | 84.7MB 5.0MB/s eta 0:00:08    71% |######################          | 86.8MB 5.9MB/s eta 0:00:06    72% |#######################         | 87.4MB 5.2MB/s eta 0:00:07    76% |########################        | 91.9MB 5.2MB/s eta 0:00:06    76% |########################        | 92.2MB 5.6MB/s eta 0:00:06    76% |########################        | 92.9MB 8.0MB/s eta 0:00:04    78% |#########################       | 94.9MB 6.9MB/s eta 0:00:04    78% |#########################       | 95.3MB 5.2MB/s eta 0:00:05    79% |#########################       | 95.7MB 6.9MB/s eta 0:00:04    82% |##########################      | 99.6MB 5.6MB/s eta 0:00:04    83% |##########################      | 100.6MB 5.7MB/s eta 0:00:04    83% |##########################      | 101.0MB 6.1MB/s eta 0:00:04    84% |##########################      | 101.7MB 6.4MB/s eta 0:00:03    85% |###########################     | 103.1MB 4.9MB/s eta 0:00:04    85% |###########################     | 103.5MB 7.5MB/s eta 0:00:03    85% |###########################     | 103.8MB 7.0MB/s eta 0:00:03    86% |###########################     | 104.1MB 7.3MB/s eta 0:00:03    87% |############################    | 105.9MB 10.5MB/s eta 0:00:02    88% |############################    | 106.6MB 5.4MB/s eta 0:00:03    89% |############################    | 108.3MB 6.4MB/s eta 0:00:02    93% |#############################   | 112.6MB 8.4MB/s eta 0:00:01    93% |#############################   | 113.0MB 6.2MB/s eta 0:00:02    93% |##############################  | 113.4MB 4.9MB/s eta 0:00:02    95% |##############################  | 115.0MB 5.2MB/s eta 0:00:02    96% |##############################  | 116.0MB 5.4MB/s eta 0:00:01    97% |############################### | 118.2MB 8.0MB/s eta 0:00:01
    [?25hInstalling collected packages: en-core-web-md
      Running setup.py install for en-core-web-md ... [?25ldone
    [?25hSuccessfully installed en-core-web-md-2.0.0
    
    [93m    Linking successful[0m
        /usr/local/lib/python3.6/dist-packages/en_core_web_md -->
        /usr/local/lib/python3.6/dist-packages/spacy/data/en_core_web_md
    
        You can now load the model via spacy.load('en_core_web_md')
    



```python
import spacy

nlp = spacy.load('en_core_web_md')
```


```python
def similar(a, b):
    """Get a similarity metric for strings a and b"""
    spacy_a = nlp(a)
    spacy_b = nlp(b)
    return spacy_a.similarity(spacy_b)
```


```python
test_random_tweets(tweets, bible_data)
```

    -----------------
    Tweet text: Next-Gen Bluetooth Bulb Controllable Via Smartphone | Freshome http://t.co/5oEjvL6c [A great patented idea - get it to market!]
    
    Nehemiah 3:32, Between the ascent of the corner and the sheep gate repaired the goldsmiths and the merchants., 0.38009049773755654
    
    Jeremiah 27:5, I have made the earth, the men and the animals that are on the surface of the earth, by my great power and by my outstretched arm; and I give it to whom it seems right to me., 0.3588039867109635
    
    1 Corinthians 10:7, Neither be idolaters, as some of them were. As it is written, The people sat down to eat and drink, and rose up to play., 0.3562753036437247
    
    Acts 2:20, The sun will be turned into darkness, and the moon into blood, before the great and glorious day of the Lord comes., 0.35537190082644626
    
    Matthew 18:4, Whoever therefore humbles himself as this little child, the same is the greatest in the Kingdom of Heaven., 0.351931330472103
    -----------------
    Tweet text: Ask Yourself: Are You Happier Now Than You Were 10 Years Ago? https://t.co/1MeEzAkHcT [+ report on happiness &amp; parenting: -ve ~ w/ GDP]
    
    Hosea 8:8, Israel is swallowed up. Now they are among the nations like a worthless thing., 0.35023041474654376
    
    Deuteronomy 29:4, but Yahweh has not given you a heart to know, and eyes to see, and ears to hear, to this day., 0.33620689655172414
    
    Amos 6:12, Do horses run on the rocky crags? Does one plow there with oxen? But you have turned justice into poison, and the fruit of righteousness into bitterness;, 0.3356164383561644
    
    Ephesians 4:28, Let him who stole steal no more; but rather let him labor, working with his hands the thing that is good, that he may have something to give to him who has need., 0.3333333333333333
    
    Psalm 112:1, Praise Yah! Blessed is the man who fears Yahweh, who delights greatly in his commandments., 0.3318777292576419
    -----------------
    Tweet text: The more possessions a person has, &amp; the more orderly the society, the greater the frequency of corporal punishment for children.
    
    1 Chronicles 26:19, These were the divisions of the doorkeepers; of the sons of the Korahites, and of the sons of Merari., 0.4700854700854701
    
    1 Samuel 17:31, When the words were heard which David spoke, they rehearsed them before Saul; and he sent for him., 0.4588744588744589
    
    2 Kings 20:4, It happened, before Isaiah had gone out into the middle part of the city, that the word of Yahweh came to him, saying,, 0.4541832669322709
    
    Exodus 39:38, the golden altar, the anointing oil, the sweet incense, the screen for the door of the Tent,, 0.4533333333333333
    
    Ezekiel 47:15, This shall be the border of the land: On the north side, from the great sea, by the way of Hethlon, to the entrance of Zedad;, 0.4496124031007752
    -----------------
    Tweet text: Rather than use subordinate clauses, old languages (~ C10 BC) juxtapose events according to temporal order.
    
    Genesis 30:34, Laban said, Behold, let it be according to your word., 0.475
    
    Numbers 15:12, According to the number that you shall prepare, so you shall do to everyone according to their number., 0.45933014354066987
    
    Numbers 26:53, To these the land shall be divided for an inheritance according to the number of names., 0.4329896907216495
    
    2 Kings 4:44, So he set it before them, and they ate, and left some of it, according to the word of Yahweh., 0.43
    
    2 Kings 7:16, The people went out, and plundered the camp of the Syrians. So a measure of fine flour was [sold] for a shekel, and two measures of barley for a shekel, according to the word of Yahweh., 0.4246575342465753
    -----------------
    Tweet text: The Economist | Amazon: http://t.co/wBm32F26yw [the world's 9th biggest retailer didn't exist 20 years ago] http://t.co/JY6Uq3iSIB
    
    2 Thessalonians 2:6, Now you know what is restraining him, to the end that he may be revealed in his own season., 0.39819004524886875
    
    Mark 16:13, They went away and told it to the rest. They didn't believe them, either., 0.39408866995073893
    
    John 14:24, He who doesn't love me doesn't keep my words. The word which you hear isn't mine, but the Father's who sent me., 0.37344398340248963
    
    1 Timothy 6:7, For we brought nothing into the world, and we certainly can't carry anything out., 0.3696682464454976
    
    Proverbs 13:1, A wise son listens to his father's instruction, but a scoffer doesn't listen to rebuke., 0.3686635944700461


### Gensim

Gensim needs a little bit of pre-processing to convert our texts into vector form. We need to get a bag of words that represents each portion of text.

First we need to tokenise our text. We can use spaCy or NLTK to do this. (The method above involves generating a spaCy doc for each Bible passage - we can maybe do this once and then use elsewhere.)

Then we filter the text and convert it into a vector form.

The procedure below mirrors the [Gensim tutorial](https://radimrehurek.com/gensim/tut1.html).


```python
# This took quite a long time so I might go for the quicker word_tokenize from nltk
# spacy_bible = [(verse, nlp(passage)) for verse, passage in bible_data]
```


```python
from nltk import word_tokenize
tokenised = [(verse, word_tokenize(passage)) for verse, passage in bible_data]
```


```python
tokenised[3]
```




    ('Genesis 1:4',
     ['God',
      'saw',
      'the',
      'light',
      ',',
      'and',
      'saw',
      'that',
      'it',
      'was',
      'good',
      '.',
      'God',
      'divided',
      'the',
      'light',
      'from',
      'the',
      'darkness',
      '.'])




```python
def process_words(tokens):
    """ Remove digits and punctuation from text and convert to lower case. """
    # Alternative for complete text is re.sub('\W+', '', text)
    return [w.lower() for w in tokens if w.isalpha()]
```


```python
tokenised = [(verse, process_words(tokens)) for verse, tokens in tokenised]
```


```python
tokenised[3]
```




    ('Genesis 1:4',
     ['god',
      'saw',
      'the',
      'light',
      'and',
      'saw',
      'that',
      'it',
      'was',
      'good',
      'god',
      'divided',
      'the',
      'light',
      'from',
      'the',
      'darkness'])




```python
texts = [tokens for _, tokens in tokenised]
```


```python
# Import NLTK modules
from nltk import word_tokenize
from nltk.corpus import stopwords
# Load stopwords
ENG_STOPWORDS = stopwords.words('english')

def text_preprocessing(original_text):
    """Clean and process texts for Gensim methods.""" 
    # Tokenise
    tokenised = word_tokenize(original_text) 
    
    # Convert to lowercase and remove non-text / stopwords
    tokenised = [w.lower() for w in tokenised if (w.isalpha() and w not in ENG_STOPWORDS)]
    return tokenised
```


```python
text_preprocessing(bible_data[3][1])
```




    ['god', 'saw', 'light', 'saw', 'good', 'god', 'divided', 'light', 'darkness']




```python
texts = [text_preprocessing(passage) for _, passage in bible_data]
```


```python
texts[5]
```




    ['god',
     'said',
     'let',
     'expanse',
     'middle',
     'waters',
     'let',
     'divide',
     'waters',
     'waters']




```python
# Create a dictionary from our processed bible texts

from gensim import corpora

# Create a dictionary that maps numbers to words
dictionary = corpora.Dictionary(texts)
# Save dictionary
dictionary.save('bible.dict')
print(dictionary)
```

    2018-06-21 12:50:20,527 : INFO : adding document #0 to Dictionary(0 unique tokens: [])
    2018-06-21 12:50:20,799 : INFO : adding document #10000 to Dictionary(6375 unique tokens: ['beginning', 'created', 'earth', 'god', 'heavens']...)
    2018-06-21 12:50:21,016 : INFO : adding document #20000 to Dictionary(9840 unique tokens: ['beginning', 'created', 'earth', 'god', 'heavens']...)
    2018-06-21 12:50:21,242 : INFO : adding document #30000 to Dictionary(12041 unique tokens: ['beginning', 'created', 'earth', 'god', 'heavens']...)
    2018-06-21 12:50:21,272 : INFO : built Dictionary(12255 unique tokens: ['beginning', 'created', 'earth', 'god', 'heavens']...) from 31102 documents (total 370556 corpus positions)
    2018-06-21 12:50:21,276 : INFO : saving Dictionary object under bible.dict, separately None
    2018-06-21 12:50:21,288 : INFO : saved bible.dict


    Dictionary(12255 unique tokens: ['beginning', 'created', 'earth', 'god', 'heavens']...)



```python
corpus = [dictionary.doc2bow(text) for text in texts]
# Save corpus for later
corpora.MmCorpus.serialize('bible.mm', corpus)
print(corpus)
```

    2018-06-21 12:50:26,077 : INFO : storing corpus in Matrix Market format to bible.mm
    2018-06-21 12:50:26,082 : INFO : saving sparse matrix to bible.mm
    2018-06-21 12:50:26,085 : INFO : PROGRESS: saving document #0
    2018-06-21 12:50:26,122 : INFO : PROGRESS: saving document #1000
    2018-06-21 12:50:26,162 : INFO : PROGRESS: saving document #2000
    2018-06-21 12:50:26,199 : INFO : PROGRESS: saving document #3000
    2018-06-21 12:50:26,237 : INFO : PROGRESS: saving document #4000
    2018-06-21 12:50:26,274 : INFO : PROGRESS: saving document #5000
    2018-06-21 12:50:26,313 : INFO : PROGRESS: saving document #6000
    2018-06-21 12:50:26,350 : INFO : PROGRESS: saving document #7000
    2018-06-21 12:50:26,391 : INFO : PROGRESS: saving document #8000
    2018-06-21 12:50:26,429 : INFO : PROGRESS: saving document #9000
    2018-06-21 12:50:26,470 : INFO : PROGRESS: saving document #10000
    2018-06-21 12:50:26,504 : INFO : PROGRESS: saving document #11000
    2018-06-21 12:50:26,549 : INFO : PROGRESS: saving document #12000
    2018-06-21 12:50:26,596 : INFO : PROGRESS: saving document #13000
    2018-06-21 12:50:26,629 : INFO : PROGRESS: saving document #14000
    2018-06-21 12:50:26,660 : INFO : PROGRESS: saving document #15000
    2018-06-21 12:50:26,695 : INFO : PROGRESS: saving document #16000
    2018-06-21 12:50:26,727 : INFO : PROGRESS: saving document #17000
    2018-06-21 12:50:26,769 : INFO : PROGRESS: saving document #18000
    2018-06-21 12:50:26,807 : INFO : PROGRESS: saving document #19000
    2018-06-21 12:50:26,858 : INFO : PROGRESS: saving document #20000
    2018-06-21 12:50:26,901 : INFO : PROGRESS: saving document #21000
    2018-06-21 12:50:26,940 : INFO : PROGRESS: saving document #22000
    2018-06-21 12:50:26,978 : INFO : PROGRESS: saving document #23000
    2018-06-21 12:50:27,011 : INFO : PROGRESS: saving document #24000
    2018-06-21 12:50:27,043 : INFO : PROGRESS: saving document #25000
    2018-06-21 12:50:27,075 : INFO : PROGRESS: saving document #26000
    2018-06-21 12:50:27,107 : INFO : PROGRESS: saving document #27000
    2018-06-21 12:50:27,141 : INFO : PROGRESS: saving document #28000
    2018-06-21 12:50:27,179 : INFO : PROGRESS: saving document #29000
    2018-06-21 12:50:27,213 : INFO : PROGRESS: saving document #30000
    2018-06-21 12:50:27,252 : INFO : PROGRESS: saving document #31000
    2018-06-21 12:50:27,259 : INFO : saved 31102x12255 matrix, density=0.089% (339121/381155010)
    2018-06-21 12:50:27,262 : INFO : saving MmCorpus index to bible.mm.index
    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.
    
    Current values:
    NotebookApp.iopub_data_rate_limit=1000000.0 (bytes/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    


On a first run of this we note that most topics are defined by common stopwords. Let's get rid of these.


```python
from gensim import models, similarities
# We'll start with LSI and a 100D vector
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=100)
```

    2018-06-21 12:51:15,457 : INFO : using serial LSI version on this node
    2018-06-21 12:51:15,458 : INFO : updating model with new documents
    2018-06-21 12:51:15,460 : INFO : preparing a new chunk of documents
    2018-06-21 12:51:15,620 : INFO : using 100 extra samples and 2 power iterations
    2018-06-21 12:51:15,626 : INFO : 1st phase: constructing (12255, 200) action matrix
    2018-06-21 12:51:15,866 : INFO : orthonormalizing (12255, 200) action matrix
    2018-06-21 12:51:16,872 : INFO : 2nd phase: running dense svd on (200, 20000) matrix
    2018-06-21 12:51:18,142 : INFO : computing the final decomposition
    2018-06-21 12:51:18,146 : INFO : keeping 100 factors (discarding 18.228% of energy spectrum)
    2018-06-21 12:51:18,326 : INFO : processed documents up to #20000
    2018-06-21 12:51:18,329 : INFO : topic #0(128.237): 0.579*"shall" + 0.450*"yahweh" + 0.419*"i" + 0.176*"said" + 0.173*"god" + 0.131*"israel" + 0.108*"king" + 0.106*"the" + 0.087*"he" + 0.085*"house"
    2018-06-21 12:51:18,333 : INFO : topic #1(94.911): -0.741*"shall" + 0.569*"i" + 0.178*"yahweh" + 0.174*"said" + 0.089*"king" + 0.081*"god" + 0.069*"israel" + -0.056*"you" + 0.049*"son" + -0.045*"offering"
    2018-06-21 12:51:18,336 : INFO : topic #2(83.248): 0.656*"i" + -0.593*"yahweh" + 0.255*"shall" + -0.159*"israel" + -0.158*"god" + -0.138*"king" + -0.103*"the" + -0.081*"house" + -0.081*"children" + -0.076*"son"
    2018-06-21 12:51:18,338 : INFO : topic #3(70.173): -0.513*"yahweh" + 0.490*"king" + 0.355*"son" + 0.249*"the" + 0.243*"said" + 0.167*"israel" + 0.139*"he" + -0.124*"i" + 0.117*"children" + 0.107*"men"
    2018-06-21 12:51:18,342 : INFO : topic #4(57.722): 0.504*"said" + -0.451*"son" + -0.379*"children" + -0.349*"israel" + 0.339*"he" + -0.131*"i" + 0.129*"let" + 0.118*"us" + 0.109*"go" + 0.108*"god"
    2018-06-21 12:51:18,345 : INFO : preparing a new chunk of documents
    2018-06-21 12:51:18,454 : INFO : using 100 extra samples and 2 power iterations
    2018-06-21 12:51:18,460 : INFO : 1st phase: constructing (12255, 200) action matrix
    2018-06-21 12:51:18,589 : INFO : orthonormalizing (12255, 200) action matrix
    2018-06-21 12:51:19,492 : INFO : 2nd phase: running dense svd on (200, 11102) matrix
    2018-06-21 12:51:20,134 : INFO : computing the final decomposition
    2018-06-21 12:51:20,138 : INFO : keeping 100 factors (discarding 19.942% of energy spectrum)
    2018-06-21 12:51:20,313 : INFO : merging projections: (12255, 100) + (12255, 100)
    2018-06-21 12:51:20,720 : INFO : keeping 100 factors (discarding 5.936% of energy spectrum)
    2018-06-21 12:51:20,956 : INFO : processed documents up to #31102
    2018-06-21 12:51:20,961 : INFO : topic #0(152.944): 0.600*"i" + 0.513*"shall" + 0.354*"yahweh" + 0.169*"said" + 0.157*"god" + 0.104*"israel" + 0.092*"the" + 0.082*"king" + 0.082*"he" + 0.075*"land"
    2018-06-21 12:51:20,965 : INFO : topic #1(116.465): -0.735*"shall" + 0.651*"i" + 0.093*"said" + -0.062*"yahweh" + -0.050*"you" + -0.049*"offering" + -0.038*"the" + -0.025*"priest" + 0.023*"know" + 0.021*"for"
    2018-06-21 12:51:20,966 : INFO : topic #2(97.426): 0.592*"yahweh" + -0.415*"i" + -0.394*"shall" + 0.258*"god" + 0.177*"said" + 0.167*"israel" + 0.167*"king" + 0.144*"the" + 0.118*"son" + 0.107*"house"
    2018-06-21 12:51:20,968 : INFO : topic #3(79.928): -0.604*"yahweh" + 0.379*"said" + 0.303*"king" + 0.284*"son" + 0.245*"the" + 0.245*"he" + 0.145*"man" + 0.144*"one" + 0.096*"came" + -0.095*"i"
    2018-06-21 12:51:20,974 : INFO : topic #4(69.851): 0.857*"god" + -0.237*"king" + -0.213*"yahweh" + -0.165*"son" + -0.162*"the" + -0.105*"israel" + 0.103*"us" + -0.093*"children" + 0.077*"for" + 0.064*"jesus"



```python
# Create index
index = similarities.MatrixSimilarity(lsi[corpus])
index.save('bible.index')
```

    2018-06-21 12:53:38,112 : WARNING : scanning corpus to determine the number of features (consider setting `num_features` explicitly)
    2018-06-21 12:53:39,874 : INFO : creating matrix with 31102 documents and 100 features
    2018-06-21 12:53:43,628 : INFO : saving MatrixSimilarity object under bible.index, separately None
    2018-06-21 12:53:43,786 : INFO : saved bible.index



```python
def text2vec(text, dictionary, lsi):
    """Convert a portion of text to an LSI vector."""
    processed = text_preprocessing(text)
    vec_bow = dictionary.doc2bow(processed)
    vec_lsi = lsi[vec_bow] # convert the query to LSI space
    return vec_lsi
```


```python
vec_lsi = text2vec(tweets[5], dictionary, lsi)
```


```python
sims = index[vec_lsi] # perform a similarity query against the corpus
print(list(enumerate(sims))[0:5]) # print (document_number, document_similarity) 2-tuples
```

    [(0, 0.026685458), (1, 0.010693545), (2, 0.02526992), (3, 0.018537477), (4, -0.0064496454)]



```python
sims_sorted = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims_sorted[0:5]) # print sorted (document number, similarity score) 2-tuples
```

    [(14471, 0.98052335), (16335, 0.78113711), (5388, 0.75273919), (17916, 0.74159586), (26053, 0.73570257)]



```python
bible_data[sims_sorted[0][0]]
```




    ('Psalm 37:21',
     "The wicked borrow, and don't pay back, but the righteous give generously.")




```python
tweets[5]
```




    '“The ungovernable metropolis, with its fluid population and ethnic and occupational enclaves, is an affront to a mindset that envisions a world of harmony, purity, and organic wholeness.” - to thrive you need to give up unattainable perfection and unquestioning agreement'



Now let's fold all this into a function.


```python
zipped = [(p, v, s) for (p, v), s in zip(bible_data, sims)]
```


```python
zipped[14471]
```




    ('Psalm 37:21',
     "The wicked borrow, and don't pay back, but the righteous give generously.",
     0.98052335)




```python
# Import gensim modules
from gensim import corpora, models, similarities

# Import NLTK modules
from nltk import word_tokenize
from nltk.corpus import stopwords
# Load stopwords
ENG_STOPWORDS = stopwords.words('english')

def text_preprocessing(original_text):
    """Clean and process texts for Gensim methods.""" 
    # Tokenise
    tokenised = word_tokenize(original_text) 
    
    # Convert to lowercase and remove non-text / stopwords
    tokenised = [w.lower() for w in tokenised if (w.isalpha() and w not in ENG_STOPWORDS)]
    return tokenised

def text2vec(text, dictionary, lsi):
    """Convert a portion of text to an LSI vector."""
    processed = text_preprocessing(text)
    vec_bow = dictionary.doc2bow(processed)
    vec_lsi = lsi[vec_bow] # convert the query to LSI space
    return vec_lsi

def build_data(tweets, bible_data):
    """Generate variables for matching."""
    # Process text
    texts = [text_preprocessing(passage) for _, passage in bible_data]
    # Build dictionary
    dictionary = corpora.Dictionary(texts)
    # Convert bible data to corpus
    corpus = [dictionary.doc2bow(text) for text in texts]
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=100)
    # Create index
    index = similarities.MatrixSimilarity(lsi[corpus])
    # Save all of these
    dictionary.save('bible.dict')
    corpora.MmCorpus.serialize('bible.mm', corpus)
    lsi.save('bible.lsi')
    index.save('bible.index')
    return dictionary, corpus, lsi, index

def get_matches(tweet, bible_data, dictionary, lsi, index):
    """Match a tweet against the bible_data."""
    # To run this we need dictionary, lsi, and index variables
    # Get matches
    vec_lsi = text2vec(tweet, dictionary, lsi)
    sims = index[vec_lsi] # perform a similarity query against the corpus
    scores = [(p, v, s) for (p, v), s in zip(bible_data, sims)]
    # Sort by descending score
    scores.sort(key=lambda tup: tup[2], reverse = True) 
    return scores

def test_random_tweets(tweets, bible_data, n=5, k=5):
    """Print n examples for k tweets selected at random."""
    try:
        dictionary = corpora.Dictionary.load('bible.dict')
        corpus = corpora.MmCorpus('bible.mm')
        lsi = models.LsiModel.load('bible.lsi')
        index = similarities.MatrixSimilarity.load('bible.index')
    except FileNotFoundError:
        dictionary, corpus, lsi, index = build_data(tweets, bible_data)
        
    import random
    num_tweets = len(tweets)
    indices = random.sample(range(0, num_tweets), k)
    for i in indices:
        tweet = tweets[i]
        print("-----------------")
        print("Tweet text: {}".format(tweet))
        scores = get_matches(tweet, bible_data, dictionary, lsi, index)
        for verse, passage, score in scores[0:n]:
            print("\n{0}, {1}, {2}".format(verse, passage, score))
```

    2018-06-23 08:15:34,349 : INFO : 'pattern' package not found; tag filters are not available for English



```python
test_random_tweets(tweets, bible_data)
```

    2018-06-21 13:41:59,527 : INFO : loading Dictionary object from bible.dict
    2018-06-21 13:41:59,541 : INFO : loaded bible.dict
    2018-06-21 13:41:59,552 : INFO : loaded corpus index from bible.mm.index
    2018-06-21 13:41:59,557 : INFO : initializing cython corpus reader from bible.mm
    2018-06-21 13:41:59,562 : INFO : accepted corpus with 31102 documents, 12255 features, 339121 non-zero entries
    2018-06-21 13:41:59,564 : INFO : loading LsiModel object from bible.lsi
    2018-06-21 13:42:09,193 : INFO : adding document #0 to Dictionary(0 unique tokens: [])
    2018-06-21 13:42:09,432 : INFO : adding document #10000 to Dictionary(6375 unique tokens: ['beginning', 'created', 'earth', 'god', 'heavens']...)
    2018-06-21 13:42:09,651 : INFO : adding document #20000 to Dictionary(9840 unique tokens: ['beginning', 'created', 'earth', 'god', 'heavens']...)
    2018-06-21 13:42:09,875 : INFO : adding document #30000 to Dictionary(12041 unique tokens: ['beginning', 'created', 'earth', 'god', 'heavens']...)
    2018-06-21 13:42:09,905 : INFO : built Dictionary(12255 unique tokens: ['beginning', 'created', 'earth', 'god', 'heavens']...) from 31102 documents (total 370556 corpus positions)
    2018-06-21 13:42:10,369 : INFO : using serial LSI version on this node
    2018-06-21 13:42:10,372 : INFO : updating model with new documents
    2018-06-21 13:42:10,375 : INFO : preparing a new chunk of documents
    2018-06-21 13:42:10,533 : INFO : using 100 extra samples and 2 power iterations
    2018-06-21 13:42:10,536 : INFO : 1st phase: constructing (12255, 200) action matrix
    2018-06-21 13:42:10,752 : INFO : orthonormalizing (12255, 200) action matrix
    2018-06-21 13:42:11,558 : INFO : 2nd phase: running dense svd on (200, 20000) matrix
    2018-06-21 13:42:12,703 : INFO : computing the final decomposition
    2018-06-21 13:42:12,707 : INFO : keeping 100 factors (discarding 18.237% of energy spectrum)
    2018-06-21 13:42:12,892 : INFO : processed documents up to #20000
    2018-06-21 13:42:12,900 : INFO : topic #0(128.237): 0.579*"shall" + 0.450*"yahweh" + 0.419*"i" + 0.176*"said" + 0.173*"god" + 0.131*"israel" + 0.108*"king" + 0.106*"the" + 0.087*"he" + 0.085*"house"
    2018-06-21 13:42:12,906 : INFO : topic #1(94.911): -0.741*"shall" + 0.569*"i" + 0.178*"yahweh" + 0.174*"said" + 0.089*"king" + 0.081*"god" + 0.069*"israel" + -0.056*"you" + 0.049*"son" + -0.045*"offering"
    2018-06-21 13:42:12,910 : INFO : topic #2(83.248): -0.656*"i" + 0.593*"yahweh" + -0.255*"shall" + 0.159*"israel" + 0.158*"god" + 0.138*"king" + 0.103*"the" + 0.081*"house" + 0.081*"children" + 0.076*"son"
    2018-06-21 13:42:12,913 : INFO : topic #3(70.173): -0.513*"yahweh" + 0.490*"king" + 0.355*"son" + 0.249*"the" + 0.243*"said" + 0.167*"israel" + 0.139*"he" + -0.124*"i" + 0.117*"children" + 0.107*"men"
    2018-06-21 13:42:12,919 : INFO : topic #4(57.722): -0.504*"said" + 0.451*"son" + 0.379*"children" + 0.350*"israel" + -0.339*"he" + 0.131*"i" + -0.129*"let" + -0.118*"us" + -0.109*"go" + -0.108*"god"
    2018-06-21 13:42:12,922 : INFO : preparing a new chunk of documents
    2018-06-21 13:42:13,026 : INFO : using 100 extra samples and 2 power iterations
    2018-06-21 13:42:13,031 : INFO : 1st phase: constructing (12255, 200) action matrix
    2018-06-21 13:42:13,162 : INFO : orthonormalizing (12255, 200) action matrix
    2018-06-21 13:42:14,004 : INFO : 2nd phase: running dense svd on (200, 11102) matrix
    2018-06-21 13:42:14,555 : INFO : computing the final decomposition
    2018-06-21 13:42:14,558 : INFO : keeping 100 factors (discarding 20.022% of energy spectrum)
    2018-06-21 13:42:14,734 : INFO : merging projections: (12255, 100) + (12255, 100)
    2018-06-21 13:42:15,064 : INFO : keeping 100 factors (discarding 5.910% of energy spectrum)
    2018-06-21 13:42:15,312 : INFO : processed documents up to #31102
    2018-06-21 13:42:15,316 : INFO : topic #0(152.944): 0.600*"i" + 0.513*"shall" + 0.354*"yahweh" + 0.169*"said" + 0.157*"god" + 0.104*"israel" + 0.092*"the" + 0.082*"king" + 0.082*"he" + 0.075*"land"
    2018-06-21 13:42:15,322 : INFO : topic #1(116.465): -0.735*"shall" + 0.651*"i" + 0.093*"said" + -0.062*"yahweh" + -0.050*"you" + -0.049*"offering" + -0.038*"the" + -0.025*"priest" + 0.023*"know" + 0.021*"for"
    2018-06-21 13:42:15,325 : INFO : topic #2(97.426): 0.592*"yahweh" + -0.415*"i" + -0.394*"shall" + 0.258*"god" + 0.177*"said" + 0.167*"israel" + 0.167*"king" + 0.144*"the" + 0.118*"son" + 0.107*"house"
    2018-06-21 13:42:15,328 : INFO : topic #3(79.927): -0.604*"yahweh" + 0.379*"said" + 0.303*"king" + 0.284*"son" + 0.245*"the" + 0.245*"he" + 0.145*"man" + 0.144*"one" + 0.096*"came" + -0.095*"i"
    2018-06-21 13:42:15,330 : INFO : topic #4(69.851): 0.857*"god" + -0.237*"king" + -0.213*"yahweh" + -0.164*"son" + -0.162*"the" + -0.105*"israel" + 0.103*"us" + -0.093*"children" + 0.077*"for" + 0.064*"jesus"
    2018-06-21 13:42:15,337 : WARNING : scanning corpus to determine the number of features (consider setting `num_features` explicitly)
    2018-06-21 13:42:16,933 : INFO : creating matrix with 31102 documents and 100 features
    2018-06-21 13:42:20,744 : INFO : saving Dictionary object under bible.dict, separately None
    2018-06-21 13:42:20,756 : INFO : saved bible.dict
    2018-06-21 13:42:20,762 : INFO : storing corpus in Matrix Market format to bible.mm
    2018-06-21 13:42:20,768 : INFO : saving sparse matrix to bible.mm
    2018-06-21 13:42:20,770 : INFO : PROGRESS: saving document #0
    2018-06-21 13:42:20,809 : INFO : PROGRESS: saving document #1000
    2018-06-21 13:42:20,846 : INFO : PROGRESS: saving document #2000
    2018-06-21 13:42:20,884 : INFO : PROGRESS: saving document #3000
    2018-06-21 13:42:20,923 : INFO : PROGRESS: saving document #4000
    2018-06-21 13:42:20,961 : INFO : PROGRESS: saving document #5000
    2018-06-21 13:42:21,002 : INFO : PROGRESS: saving document #6000
    2018-06-21 13:42:21,041 : INFO : PROGRESS: saving document #7000
    2018-06-21 13:42:21,084 : INFO : PROGRESS: saving document #8000
    2018-06-21 13:42:21,125 : INFO : PROGRESS: saving document #9000
    2018-06-21 13:42:21,167 : INFO : PROGRESS: saving document #10000
    2018-06-21 13:42:21,203 : INFO : PROGRESS: saving document #11000
    2018-06-21 13:42:21,245 : INFO : PROGRESS: saving document #12000
    2018-06-21 13:42:21,286 : INFO : PROGRESS: saving document #13000
    2018-06-21 13:42:21,314 : INFO : PROGRESS: saving document #14000
    2018-06-21 13:42:21,345 : INFO : PROGRESS: saving document #15000
    2018-06-21 13:42:21,377 : INFO : PROGRESS: saving document #16000
    2018-06-21 13:42:21,405 : INFO : PROGRESS: saving document #17000
    2018-06-21 13:42:21,448 : INFO : PROGRESS: saving document #18000
    2018-06-21 13:42:21,488 : INFO : PROGRESS: saving document #19000
    2018-06-21 13:42:21,530 : INFO : PROGRESS: saving document #20000
    2018-06-21 13:42:21,583 : INFO : PROGRESS: saving document #21000
    2018-06-21 13:42:21,627 : INFO : PROGRESS: saving document #22000
    2018-06-21 13:42:21,680 : INFO : PROGRESS: saving document #23000
    2018-06-21 13:42:21,727 : INFO : PROGRESS: saving document #24000
    2018-06-21 13:42:21,762 : INFO : PROGRESS: saving document #25000
    2018-06-21 13:42:21,795 : INFO : PROGRESS: saving document #26000
    2018-06-21 13:42:21,827 : INFO : PROGRESS: saving document #27000
    2018-06-21 13:42:21,868 : INFO : PROGRESS: saving document #28000
    2018-06-21 13:42:21,903 : INFO : PROGRESS: saving document #29000
    2018-06-21 13:42:21,935 : INFO : PROGRESS: saving document #30000
    2018-06-21 13:42:21,970 : INFO : PROGRESS: saving document #31000
    2018-06-21 13:42:21,977 : INFO : saved 31102x12255 matrix, density=0.089% (339121/381155010)
    2018-06-21 13:42:21,980 : INFO : saving MmCorpus index to bible.mm.index
    2018-06-21 13:42:21,985 : INFO : saving Projection object under bibl.lsi.projection, separately None
    2018-06-21 13:42:22,092 : INFO : saved bibl.lsi.projection
    2018-06-21 13:42:22,096 : INFO : saving LsiModel object under bibl.lsi, separately None
    2018-06-21 13:42:22,098 : INFO : not storing attribute projection
    2018-06-21 13:42:22,101 : INFO : not storing attribute dispatcher
    2018-06-21 13:42:22,110 : INFO : saved bibl.lsi
    2018-06-21 13:42:22,115 : INFO : saving MatrixSimilarity object under bible.index, separately None
    2018-06-21 13:42:22,250 : INFO : saved bible.index


    -----------------
    Tweet text: @sustrans Kids to Newbridge primary in Bath have river cycle path close - but no safe way to travel 200m up hill & across A-road or to path
    
    Genesis 49:17, Dan will be a serpent in the way, an adder in the path, That bites the horse's heels, so that his rider falls backward., 0.9940089583396912
    
    Psalm 80:12, Why have you broken down its walls, so that all those who pass by the way pluck it?, 0.9881158471107483
    
    Proverbs 13:6, Righteousness guards the way of integrity, but wickedness overthrows the sinner., 0.9859569668769836
    
    Genesis 35:19, Rachel died, and was buried in the way to Ephrath (the same is Bethlehem)., 0.9790574312210083
    
    Ezekiel 12:5, Dig through the wall in their sight, and carry your stuff out that way., 0.9741089344024658
    -----------------
    Tweet text: Stand-Up Comics Have to Censor Their Jokes on (US) College Campuses - The Atlantic http://t.co/W998v8oahs
    
    Lamentations 5:16, The crown is fallen from our head: Woe to us! for we have sinned., 0.9783570766448975
    
    Acts 28:2, The natives showed us uncommon kindness; for they kindled a fire, and received us all, because of the present rain, and because of the cold., 0.9126466512680054
    
    Deuteronomy 26:6, The Egyptians dealt ill with us, and afflicted us, and laid on us hard bondage:, 0.89407879114151
    
    Ezra 4:18, The letter which you sent to us has been plainly read before me., 0.863696277141571
    
    Judges 9:12, The trees said to the vine, 'Come and reign over us.', 0.8148699998855591
    -----------------
    Tweet text: Ha - was just reminded of the 90s Internet time limits, e.g. 5 hours online per week. Could do with that now.
    
    Ecclesiastes 3:4, a time to weep, and a time to laugh; a time to mourn, and a time to dance;, 0.9986615180969238
    
    Ecclesiastes 3:3, a time to kill, and a time to heal; a time to break down, and a time to build up;, 0.9981067180633545
    
    Matthew 26:16, From that time he sought opportunity to betray him., 0.9979551434516907
    
    Ecclesiastes 3:2, a time to be born, and a time to die; a time to plant, and a time to pluck up that which is planted;, 0.9977869987487793
    
    Hebrews 9:10, being only (with meats and drinks and various washings) fleshly ordinances, imposed until a time of reformation., 0.9975525736808777
    -----------------
    Tweet text: Here's What Happened To All 53 of Marissa Mayer's Yahoo Acquisitions https://t.co/0YT2TXncHN
    
    Luke 24:51, It happened, while he blessed them, that he withdrew from them, and was carried up into heaven., 0.8907666802406311
    
    Joshua 5:8, It happened, when they were done circumcising all the nation, that they stayed in their places in the camp until they were healed., 0.8737368583679199
    
    1 Kings 15:21, It happened, when Baasha heard of it, that he left off building Ramah, and lived in Tirzah., 0.8584436178207397
    
    Joshua 19:26, Allammelech, Amad, Mishal. It reached to Carmel westward, and to Shihorlibnath., 0.8564523458480835
    
    2 Corinthians 9:1, It is indeed unnecessary for me to write to you concerning the service to the saints,, 0.852521538734436
    -----------------
    Tweet text: Even with more complex deep architectures you can get a surprising amount done with simple ngram approaches - a future in hybrids? https://t.co/jD3pIhnIbP
    
    Daniel 9:5, we have sinned, and have dealt perversely, and have done wickedly, and have rebelled, even turning aside from your precepts and from your ordinances;, 0.9942108988761902
    
    Ephesians 5:11, Have no fellowship with the unfruitful works of darkness, but rather even reprove them., 0.9036127924919128
    
    Joshua 10:41, Joshua struck them from Kadesh Barnea even to Gaza, and all the country of Goshen, even to Gibeon., 0.885510265827179
    
    Numbers 21:30, We have shot at them. Heshbon has perished even to Dibon. We have laid waste even to Nophah, Which reaches to Medeba., 0.8847762942314148
    
    Deuteronomy 4:48, from Aroer, which is on the edge of the valley of the Arnon, even to Mount Sion (the same is Hermon),, 0.882093071937561


### Comparing Approaches

To compare the approaches, let's generate 200 random examples and take the top match for each of the three techniques. We will then manually score each match on a scale of 0 to 5 where 0 = no match and 5 = perfect match. Then we can see which technique comes up on top.

The easiest way to quickly compare the results is to export to a spreadsheet, with columns for the scores of each.


```python
def get_difflib_matches(tweet, bible_data):
    """Match a tweet against the bible_data."""
    # Get matches
    scores = [
        (verse, passage, SequenceMatcher(None, tweet, passage).ratio()) 
        for verse, passage in bible_data
    ]
    # Sort by descending score
    scores.sort(key=lambda tup: tup[2], reverse = True) 
    return scores

def get_spacy_matches(spacy_tweet, spacy_bible):
    """Perform matches on text as spacy docs"""
    # Get matches
    scores = [
        (verse, passage, spacy_tweet.similarity(passage)) 
        for verse, passage in spacy_bible
    ]
    # Sort by descending score
    scores.sort(key=lambda tup: tup[2], reverse = True) 
    return scores

def get_gensim_matches(tweet, bible_data, dictionary, lsi, index):
    """Match a tweet against the bible_data."""
    # To run this we need dictionary, lsi, and index variables
    # Get matches
    vec_lsi = text2vec(tweet, dictionary, lsi)
    sims = index[vec_lsi] # perform a similarity query against the corpus
    scores = [(v, p, s) for (v, p), s in zip(bible_data, sims)]
    # Sort by descending score
    scores.sort(key=lambda tup: tup[2], reverse = True) 
    return scores
```


```python
spacy_bible = [(verse, nlp(passage)) for verse, passage in bible_data]
```


```python
dictionary, corpus, lsi, index = build_data(tweets, bible_data)
```

    2018-06-23 09:02:25,811 : INFO : adding document #0 to Dictionary(0 unique tokens: [])
    2018-06-23 09:02:26,052 : INFO : adding document #10000 to Dictionary(6375 unique tokens: ['beginning', 'created', 'earth', 'god', 'heavens']...)
    2018-06-23 09:02:26,272 : INFO : adding document #20000 to Dictionary(9840 unique tokens: ['beginning', 'created', 'earth', 'god', 'heavens']...)
    2018-06-23 09:02:26,491 : INFO : adding document #30000 to Dictionary(12041 unique tokens: ['beginning', 'created', 'earth', 'god', 'heavens']...)
    2018-06-23 09:02:26,516 : INFO : built Dictionary(12255 unique tokens: ['beginning', 'created', 'earth', 'god', 'heavens']...) from 31102 documents (total 370556 corpus positions)
    2018-06-23 09:02:26,953 : INFO : using serial LSI version on this node
    2018-06-23 09:02:26,954 : INFO : updating model with new documents
    2018-06-23 09:02:26,955 : INFO : preparing a new chunk of documents
    2018-06-23 09:02:27,111 : INFO : using 100 extra samples and 2 power iterations
    2018-06-23 09:02:27,112 : INFO : 1st phase: constructing (12255, 200) action matrix
    2018-06-23 09:02:27,349 : INFO : orthonormalizing (12255, 200) action matrix
    2018-06-23 09:02:28,118 : INFO : 2nd phase: running dense svd on (200, 20000) matrix
    2018-06-23 09:02:29,147 : INFO : computing the final decomposition
    2018-06-23 09:02:29,161 : INFO : keeping 100 factors (discarding 18.228% of energy spectrum)
    2018-06-23 09:02:29,352 : INFO : processed documents up to #20000
    2018-06-23 09:02:29,357 : INFO : topic #0(128.237): 0.579*"shall" + 0.450*"yahweh" + 0.419*"i" + 0.176*"said" + 0.173*"god" + 0.131*"israel" + 0.108*"king" + 0.106*"the" + 0.087*"he" + 0.085*"house"
    2018-06-23 09:02:29,359 : INFO : topic #1(94.911): 0.741*"shall" + -0.569*"i" + -0.178*"yahweh" + -0.174*"said" + -0.089*"king" + -0.081*"god" + -0.069*"israel" + 0.056*"you" + -0.049*"son" + 0.045*"offering"
    2018-06-23 09:02:29,361 : INFO : topic #2(83.248): -0.656*"i" + 0.593*"yahweh" + -0.255*"shall" + 0.159*"israel" + 0.158*"god" + 0.138*"king" + 0.103*"the" + 0.081*"house" + 0.081*"children" + 0.076*"son"
    2018-06-23 09:02:29,363 : INFO : topic #3(70.173): 0.513*"yahweh" + -0.490*"king" + -0.355*"son" + -0.249*"the" + -0.243*"said" + -0.167*"israel" + -0.139*"he" + 0.124*"i" + -0.117*"children" + -0.107*"men"
    2018-06-23 09:02:29,365 : INFO : topic #4(57.722): 0.504*"said" + -0.451*"son" + -0.379*"children" + -0.350*"israel" + 0.339*"he" + -0.131*"i" + 0.129*"let" + 0.118*"us" + 0.109*"go" + 0.108*"god"
    2018-06-23 09:02:29,367 : INFO : preparing a new chunk of documents
    2018-06-23 09:02:29,467 : INFO : using 100 extra samples and 2 power iterations
    2018-06-23 09:02:29,469 : INFO : 1st phase: constructing (12255, 200) action matrix
    2018-06-23 09:02:29,587 : INFO : orthonormalizing (12255, 200) action matrix
    2018-06-23 09:02:30,386 : INFO : 2nd phase: running dense svd on (200, 11102) matrix
    2018-06-23 09:02:30,731 : INFO : computing the final decomposition
    2018-06-23 09:02:30,732 : INFO : keeping 100 factors (discarding 19.942% of energy spectrum)
    2018-06-23 09:02:30,935 : INFO : merging projections: (12255, 100) + (12255, 100)
    2018-06-23 09:02:31,243 : INFO : keeping 100 factors (discarding 5.902% of energy spectrum)
    2018-06-23 09:02:31,483 : INFO : processed documents up to #31102
    2018-06-23 09:02:31,485 : INFO : topic #0(152.944): 0.600*"i" + 0.513*"shall" + 0.354*"yahweh" + 0.169*"said" + 0.157*"god" + 0.104*"israel" + 0.092*"the" + 0.082*"king" + 0.082*"he" + 0.075*"land"
    2018-06-23 09:02:31,486 : INFO : topic #1(116.465): -0.735*"shall" + 0.651*"i" + 0.093*"said" + -0.062*"yahweh" + -0.050*"you" + -0.049*"offering" + -0.038*"the" + -0.025*"priest" + 0.023*"know" + 0.021*"for"
    2018-06-23 09:02:31,487 : INFO : topic #2(97.426): 0.592*"yahweh" + -0.415*"i" + -0.394*"shall" + 0.258*"god" + 0.177*"said" + 0.167*"israel" + 0.167*"king" + 0.144*"the" + 0.118*"son" + 0.107*"house"
    2018-06-23 09:02:31,489 : INFO : topic #3(79.928): -0.604*"yahweh" + 0.379*"said" + 0.303*"king" + 0.284*"son" + 0.245*"the" + 0.245*"he" + 0.145*"man" + 0.144*"one" + 0.096*"came" + -0.095*"i"
    2018-06-23 09:02:31,491 : INFO : topic #4(69.852): 0.857*"god" + -0.237*"king" + -0.213*"yahweh" + -0.165*"son" + -0.162*"the" + -0.105*"israel" + 0.103*"us" + -0.093*"children" + 0.077*"for" + 0.064*"jesus"
    2018-06-23 09:02:31,492 : WARNING : scanning corpus to determine the number of features (consider setting `num_features` explicitly)
    2018-06-23 09:02:32,948 : INFO : creating matrix with 31102 documents and 100 features
    2018-06-23 09:02:36,645 : INFO : saving Dictionary object under bible.dict, separately None
    2018-06-23 09:02:36,713 : INFO : saved bible.dict
    2018-06-23 09:02:36,714 : INFO : storing corpus in Matrix Market format to bible.mm
    2018-06-23 09:02:36,715 : INFO : saving sparse matrix to bible.mm
    2018-06-23 09:02:36,716 : INFO : PROGRESS: saving document #0
    2018-06-23 09:02:36,753 : INFO : PROGRESS: saving document #1000
    2018-06-23 09:02:36,789 : INFO : PROGRESS: saving document #2000
    2018-06-23 09:02:36,832 : INFO : PROGRESS: saving document #3000
    2018-06-23 09:02:36,863 : INFO : PROGRESS: saving document #4000
    2018-06-23 09:02:36,898 : INFO : PROGRESS: saving document #5000
    2018-06-23 09:02:36,933 : INFO : PROGRESS: saving document #6000
    2018-06-23 09:02:36,970 : INFO : PROGRESS: saving document #7000
    2018-06-23 09:02:37,005 : INFO : PROGRESS: saving document #8000
    2018-06-23 09:02:37,041 : INFO : PROGRESS: saving document #9000
    2018-06-23 09:02:37,081 : INFO : PROGRESS: saving document #10000
    2018-06-23 09:02:37,123 : INFO : PROGRESS: saving document #11000
    2018-06-23 09:02:37,169 : INFO : PROGRESS: saving document #12000
    2018-06-23 09:02:37,200 : INFO : PROGRESS: saving document #13000
    2018-06-23 09:02:37,224 : INFO : PROGRESS: saving document #14000
    2018-06-23 09:02:37,251 : INFO : PROGRESS: saving document #15000
    2018-06-23 09:02:37,277 : INFO : PROGRESS: saving document #16000
    2018-06-23 09:02:37,311 : INFO : PROGRESS: saving document #17000
    2018-06-23 09:02:37,350 : INFO : PROGRESS: saving document #18000
    2018-06-23 09:02:37,386 : INFO : PROGRESS: saving document #19000
    2018-06-23 09:02:37,421 : INFO : PROGRESS: saving document #20000
    2018-06-23 09:02:37,453 : INFO : PROGRESS: saving document #21000
    2018-06-23 09:02:37,488 : INFO : PROGRESS: saving document #22000
    2018-06-23 09:02:37,522 : INFO : PROGRESS: saving document #23000
    2018-06-23 09:02:37,563 : INFO : PROGRESS: saving document #24000
    2018-06-23 09:02:37,591 : INFO : PROGRESS: saving document #25000
    2018-06-23 09:02:37,621 : INFO : PROGRESS: saving document #26000
    2018-06-23 09:02:37,648 : INFO : PROGRESS: saving document #27000
    2018-06-23 09:02:37,680 : INFO : PROGRESS: saving document #28000
    2018-06-23 09:02:37,719 : INFO : PROGRESS: saving document #29000
    2018-06-23 09:02:37,758 : INFO : PROGRESS: saving document #30000
    2018-06-23 09:02:37,790 : INFO : PROGRESS: saving document #31000
    2018-06-23 09:02:37,795 : INFO : saved 31102x12255 matrix, density=0.089% (339121/381155010)
    2018-06-23 09:02:37,797 : INFO : saving MmCorpus index to bible.mm.index
    2018-06-23 09:02:37,799 : INFO : saving Projection object under bible.lsi.projection, separately None
    2018-06-23 09:02:37,944 : INFO : saved bible.lsi.projection
    2018-06-23 09:02:37,945 : INFO : saving LsiModel object under bible.lsi, separately None
    2018-06-23 09:02:37,946 : INFO : not storing attribute projection
    2018-06-23 09:02:37,947 : INFO : not storing attribute dispatcher
    2018-06-23 09:02:37,953 : INFO : saved bible.lsi
    2018-06-23 09:02:37,954 : INFO : saving MatrixSimilarity object under bible.index, separately None
    2018-06-23 09:02:38,053 : INFO : saved bible.index



```python
import random

rows = []
k = 200
num_tweets = len(tweets)
indices = random.sample(range(0, num_tweets), k)
for i in indices:
    tweet = tweets[i]
    dl_match = get_difflib_matches(tweet, bible_data)[0]
    spacy_tweet = nlp(tweet)
    sp_match = get_spacy_matches(spacy_tweet, spacy_bible)[0]
    gs_match = get_gensim_matches(tweet, bible_data, dictionary, lsi, index)[0]
    comparison_row = (
        tweet, 
        dl_match[1], dl_match[2], "", 
        sp_match[1].text, sp_match[2], "", 
        gs_match[1], gs_match[2], ""
    )
    rows.append(comparison_row)
```


```python
import pickle
with open("rows.pkl", 'wb') as f:
    pickle.dump(rows, f)
```


```python
import pandas as pd

row_df = pd.DataFrame(rows)
```


```python
del rows
```


```python
row_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Also: if your patent specification does not co...</td>
      <td>Therefore you need to be in subjection, not on...</td>
      <td>0.420168</td>
      <td></td>
      <td>But whatever has a blemish, that you shall not...</td>
      <td>0.921702</td>
      <td></td>
      <td>There is one body, and one Spirit, even as you...</td>
      <td>0.836346</td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>The CImg Library - C++ Template Image Processi...</td>
      <td>the people which I formed for myself, that the...</td>
      <td>0.374384</td>
      <td></td>
      <td>For the house he made windows of fixed lattice...</td>
      <td>0.736559</td>
      <td></td>
      <td>The sound of a cry from Horonaim, desolation a...</td>
      <td>0.971526</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>Criteria for boilerplate legal text: it should...</td>
      <td>He began to build in the second [day] of the s...</td>
      <td>0.447368</td>
      <td></td>
      <td>It's also written in your law that the testimo...</td>
      <td>0.923304</td>
      <td></td>
      <td>the two pillars, and the two bowls of the capi...</td>
      <td>0.711855</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hadn't realised: the difference between deduct...</td>
      <td>and the avenger of blood find him outside of t...</td>
      <td>0.386946</td>
      <td></td>
      <td>However in the assembly I would rather speak f...</td>
      <td>0.927348</td>
      <td></td>
      <td>Should he reason with unprofitable talk, or wi...</td>
      <td>0.997764</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>You need to be wrong on certain things in orde...</td>
      <td>No eye pitied you, to do any of these things t...</td>
      <td>0.452830</td>
      <td></td>
      <td>Be of the same mind one toward another. Don't ...</td>
      <td>0.966708</td>
      <td></td>
      <td>You are witnesses of these things.</td>
      <td>0.823695</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>



```python
!pip3 install xlwt
```

    Collecting xlwt
    [?25l  Downloading https://files.pythonhosted.org/packages/44/48/def306413b25c3d01753603b1a222a011b8621aed27cd7f89cbc27e6b0f4/xlwt-1.3.0-py2.py3-none-any.whl (99kB)
    [K    100% |################################| 102kB 2.9MB/s a 0:00:011
    [?25hInstalling collected packages: xlwt
    Successfully installed xlwt-1.3.0



```python

row_df.to_excel("comparison.xls")
```


