Title: 2. Tweet2Bible - Exploring the Data
Tags: preparing_data
Authors: Ben Hoyle
Summary: This post explores the data for our Tweet2Bible experiments.

# 2. Tweet2Bible - Exploring the Data

In the previous post we looked at the problem for our current project: how to match a tweet to passages of the Bible.

In this post we will look at the steps required to prepare some data for our algorithms. We will also look at the nature of the data, which may give us some insights as to how we transform our data in later stages.

## Getting the Data

### Tweets

Twitter offers an option in the settings to download your tweet archive. Log in to the web version, goto account settings and click the "Download Archive" button. You will then be sent an email with a link to the data.

Fair play to Twitter, the archive is quite cool. You get a CSV file with your tweets and some other useful data, plus a JSON archive (which can be viewed via a local HTML file). To keep things simple we'll just use the CSV file for now.


```python
import csv
with open('tweets.csv', newline='', encoding='utf-8') as csvfile:
    casereader = csv.reader(csvfile, delimiter=',')
    data = [row for row in casereader]
```


```python
data[0]
```




    ['tweet_id',
     'in_reply_to_status_id',
     'in_reply_to_user_id',
     'timestamp',
     'source',
     'text',
     'retweeted_status_id',
     'retweeted_status_user_id',
     'retweeted_status_timestamp',
     'expanded_urls']




```python
data[1:3]
```




    [['1009719610910957568',
      '1009527946036613121',
      '20524211',
      '2018-06-21 08:48:09 +0000',
      '<a href="http://twitter.com" rel="nofollow">Twitter Web Client</a>',
      "@patently It's okay - whatever photo storage app you use is already plugged into the system.",
      '',
      '',
      '',
      ''],
     ['1009409717205168128',
      '',
      '',
      '2018-06-20 12:16:45 +0000',
      '<a href="http://twitter.com/download/iphone" rel="nofollow">Twitter for iPhone</a>',
      'I love all the people the computer said “looked a bit like Rick Astley” including, I think, JFK, Jesus and Norman Bates. https://t.co/HHI0HFU0Cy',
      '',
      '',
      '',
      'https://twitter.com/quasimondo/status/1009369380042485760']]



For now we will just extract the text to get a list of strings.


```python
D1 = [d[5] for d in data[1:]]
```


```python
D1[0:5]
```




    ["@patently It's okay - whatever photo storage app you use is already plugged into the system.",
     'I love all the people the computer said “looked a bit like Rick Astley” including, I think, JFK, Jesus and Norman Bates. https://t.co/HHI0HFU0Cy',
     "“These were ancient engineers with a genius that allowed people to walk multi-tonne statues and roll multi-tonne hats - which teaches us about the society's investment in honouring their ancestors. It's quite a remarkable accomplishment” https://t.co/DzscvwJ0do https://t.co/Q7B8ioRS0h",
     'KPMG audit work unacceptable - watchdog https://t.co/9np6lWkHTG [Average remuneration per partner in 2016 = £582k]',
     'Ooo Le Sud by Nino Ferrer - well done @SpotifyUK algorithms https://t.co/ANxMxVWiJV [Question is do I prefer the original French or Nino’s English version? Also check out the brilliantly proggy Métronomie]']




```python
"We have {0} tweets.".format(len(D1))
```




    'We have 9806 tweets.'



Looking at some of our tweets, we need to unescape the text such that "&amp;" is converted to "&". This can be performed using the html library.


```python
import html

D1 = [html.unescape(t) for t in D1]
```

### Bible

The Bible is actually quite a good source of text for natural language processing projects. 

* it is free; 
* people want to make it easy to distribute; 
* it is naturally broken down into short passages; and
* it contains a variety of styles (I like to thing of it as a 2000 year old Wikipedia for middle-eastern farmers).

For this project I went to [BibleHub.net](http://biblehub.net/database/) which offers an Excel spreadsheet featuring 10 different versions of the Bible, where each row is a different verse. You get a free username and password in exchange for registration using an email address.

We can use Pandas to convert the spreadsheet into useful Python data. We then need to pick a Bible to use. I think the most modern translation will probably be best. 


```python
import pandas as pd
```


```python
# Pandas needs the xlrd package to read excel files
!pip3 install xlrd
```

    Collecting xlrd
    [?25l  Downloading https://files.pythonhosted.org/packages/07/e6/e95c4eec6221bfd8528bcc4ea252a850bffcc4be88ebc367e23a1a84b0bb/xlrd-1.1.0-py2.py3-none-any.whl (108kB)
    [K    100% |################################| 112kB 1.1MB/s ta 0:00:01   65% |#####################           | 71kB 2.0MB/s eta 0:00:01
    [?25hInstalling collected packages: xlrd
    Successfully installed xlrd-1.1.0



```python
file = 'bibles.xls'
df = pd.read_excel(file)
```


```python
df.head()
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
      <th>Verse</th>
      <th>King James Bible</th>
      <th>American Standard Version</th>
      <th>Douay-Rheims Bible</th>
      <th>Darby Bible Translation</th>
      <th>English Revised Version</th>
      <th>Webster Bible Translation</th>
      <th>World English Bible</th>
      <th>Young's Literal Translation</th>
      <th>American King James Version</th>
      <th>Weymouth New Testament</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Genesis 1:1</td>
      <td>In the beginning God created the heaven and th...</td>
      <td>In the beginning God created the heavens and t...</td>
      <td>In the beginning God created heaven, and earth.</td>
      <td>In the beginning God created the heavens and t...</td>
      <td>In the beginning God created the heaven and th...</td>
      <td>In the beginning God created the heaven and th...</td>
      <td>In the beginning God created the heavens and t...</td>
      <td>In the beginning of God's preparing the heaven...</td>
      <td>In the beginning God created the heaven and th...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Genesis 1:2</td>
      <td>And the earth was without form, and void; and ...</td>
      <td>And the earth was waste and void; and darkness...</td>
      <td>And the earth was void and empty, and darkness...</td>
      <td>And the earth was waste and empty, and darknes...</td>
      <td>And the earth was waste and void; and darkness...</td>
      <td>And the earth was without form, and void; and ...</td>
      <td>Now the earth was formless and empty. Darkness...</td>
      <td>the earth hath existed waste and void, and dar...</td>
      <td>And the earth was without form, and void; and ...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Genesis 1:3</td>
      <td>And God said, Let there be light: and there wa...</td>
      <td>And God said, Let there be light: and there wa...</td>
      <td>And God said: Be light made. And light was made.</td>
      <td>And God said, Let there be light. And there wa...</td>
      <td>And God said, Let there be light: and there wa...</td>
      <td>And God said, Let there be light: and there wa...</td>
      <td>God said, |Let there be light,| and there was ...</td>
      <td>and God saith, 'Let light be;' and light is.</td>
      <td>And God said, Let there be light: and there wa...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Genesis 1:4</td>
      <td>And God saw the light, that &lt;i&gt;it was&lt;/i&gt; good...</td>
      <td>And God saw the light, that it was good: and G...</td>
      <td>And God saw the light that it was good; and he...</td>
      <td>And God saw the light that it was good; and Go...</td>
      <td>And God saw the light, that it was good: and G...</td>
      <td>And God saw the light, that it was good: and G...</td>
      <td>God saw the light, and saw that it was good. G...</td>
      <td>And God seeth the light that it is good, and G...</td>
      <td>And God saw the light, that it was good: and G...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Genesis 1:5</td>
      <td>And God called the light Day, and the darkness...</td>
      <td>And God called the light Day, and the darkness...</td>
      <td>And he called the light Day, and the darkness ...</td>
      <td>And God called the light Day, and the darkness...</td>
      <td>And God called the light Day, and the darkness...</td>
      <td>And God called the light Day, and the darkness...</td>
      <td>God called the light |day,| and the darkness h...</td>
      <td>and God calleth to the light 'Day,' and to the...</td>
      <td>And God called the light Day, and the darkness...</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
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
      <th>Verse</th>
      <th>King James Bible</th>
      <th>American Standard Version</th>
      <th>Douay-Rheims Bible</th>
      <th>Darby Bible Translation</th>
      <th>English Revised Version</th>
      <th>Webster Bible Translation</th>
      <th>World English Bible</th>
      <th>Young's Literal Translation</th>
      <th>American King James Version</th>
      <th>Weymouth New Testament</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>31102</td>
      <td>31102</td>
      <td>31100</td>
      <td>31092</td>
      <td>31099</td>
      <td>31086</td>
      <td>31102</td>
      <td>31098</td>
      <td>31102</td>
      <td>31102</td>
      <td>7924</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>31102</td>
      <td>30840</td>
      <td>30716</td>
      <td>30886</td>
      <td>30722</td>
      <td>30687</td>
      <td>30855</td>
      <td>30776</td>
      <td>30861</td>
      <td>30825</td>
      <td>7913</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Proverbs 26:24</td>
      <td>And the LORD spake unto Moses, saying,</td>
      <td>And Jehovah spake unto Moses, saying,</td>
      <td>And the Lord spoke to Moses, saying:</td>
      <td>And Jehovah spoke to Moses, saying,</td>
      <td>And the LORD spake unto Moses, saying,</td>
      <td>And the LORD spoke to Moses, saying,</td>
      <td>Yahweh spoke to Moses, saying,</td>
      <td>And Jehovah speaketh unto Moses, saying,</td>
      <td>And the LORD spoke to Moses, saying,</td>
      <td>May grace and peace be granted to you from God...</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>72</td>
      <td>72</td>
      <td>55</td>
      <td>72</td>
      <td>72</td>
      <td>72</td>
      <td>71</td>
      <td>73</td>
      <td>72</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



Bibles!

Cool. I think the World English Bible looks good for a modern translation. It does have some annoying "|" we might want to scrub out.


```python
worldbible = df[['Verse', 'World English Bible']]
```


```python
worldbible.head()
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
      <th>Verse</th>
      <th>World English Bible</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Genesis 1:1</td>
      <td>In the beginning God created the heavens and t...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Genesis 1:2</td>
      <td>Now the earth was formless and empty. Darkness...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Genesis 1:3</td>
      <td>God said, |Let there be light,| and there was ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Genesis 1:4</td>
      <td>God saw the light, and saw that it was good. G...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Genesis 1:5</td>
      <td>God called the light |day,| and the darkness h...</td>
    </tr>
  </tbody>
</table>
</div>




```python
D2 = [tuple(x) for x in worldbible.to_records(index=False)]
```


```python
D2[0:5]
```




    [('Genesis 1:1', 'In the beginning God created the heavens and the earth.'),
     ('Genesis 1:2',
      "Now the earth was formless and empty. Darkness was on the surface of the deep. God's Spirit was hovering over the surface of the waters."),
     ('Genesis 1:3', 'God said, |Let there be light,| and there was light.'),
     ('Genesis 1:4',
      'God saw the light, and saw that it was good. God divided the light from the darkness.'),
     ('Genesis 1:5',
      'God called the light |day,| and the darkness he called |night.| There was evening and there was morning, one day.')]




```python
# Just get rid of those annoying |
D2 = [(str(v), str(t).replace("|","")) for v,t in D2]
D2[0:5]
```




    [('Genesis 1:1', 'In the beginning God created the heavens and the earth.'),
     ('Genesis 1:2',
      "Now the earth was formless and empty. Darkness was on the surface of the deep. God's Spirit was hovering over the surface of the waters."),
     ('Genesis 1:3', 'God said, Let there be light, and there was light.'),
     ('Genesis 1:4',
      'God saw the light, and saw that it was good. God divided the light from the darkness.'),
     ('Genesis 1:5',
      'God called the light day, and the darkness he called night. There was evening and there was morning, one day.')]




```python
"We have {0} Bible passages.".format(len(D2))
```




    'We have 31102 Bible passages.'



So now we D1, a set of tweets, and D2, a set of Bible passages. Let's get matching!


```python
# Let's save our data so we can easily load it in a future session
save_data = (D1, D2)
import pickle
with open("processed_data.pkl", 'wb') as f:
    pickle.dump(save_data, f)
```
