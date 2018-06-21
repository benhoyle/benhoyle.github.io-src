Title: 1. Tweet2Bible - Defining the Problem
Tags: defining_the_problem
Authors: Ben Hoyle
Summary: This post looks at how we define the problem of text similarity.

# 1. Tweet2Bible - Defining the Problem

This post looks at how we define the problem of matching Bible passages to tweets.

-----

# What is the problem?

## Problem Introduction

I need to research text matching algorithms. We start with two corpora of text data. For a portion of text in the first corpus we want to find the "most similar" portion of text from the second corpus.

For fun I decided to use two readily available corpora of short text portions: my Tweet archive and the Bible.

The devil (might have to be careful with that phrase) is in the detail of course. A key problem is how we define "most similar".

The problem can be seen as one of information retrieval.

## Formalised

We have two corpora: D1 and D2. Each corpora contains a set of text portions: ti, where i is the ith text portion. For a given text portion in D1, tg, we want to return a ranked list of text portions from D2, r.

We can define a similarity measure, s, that is a normalised metric with values from 0 to 1 that represents similarity of two text portions t1 and t2, where 0 means no similarity and 1 means t1=t2. The ranked list r thus can contain tuples (tj, sj) sorted in descending order based on sj.

We want to design a similarity function that matches our natural feeling of relevance. This is the hard part. Relevance will need to be scored subjectively, by comparatively rating different functions using human judgement (e.g. through looking at the actual results).

## Assumptions

Below are some assumptions that may apply to the problem.

- We assume that there is some underlying human "relevance", i.e. that a person plucked off the street could say, relatively that one passage is more relevant than another passage.
- We assume that similarity in some way may be reduced to a single metric between 0 and 1.


---

## Similar Problems

### Information Retrieval

The current problem is really an [information retrieval](https://en.wikipedia.org/wiki/Information_retrieval) problem. We have a source document, which is used to query a corpus and to retrieve a set of "relevant" documents. Many information retrieval systems take a set of keywords and then query an indexed database using the keywords. Google uses the PageRank algorithm and other magic to return relevant search results that are ranked based on links in and out of the web-page.

---

# Why does the problem need to be solved?

There are many areas where it would be useful to match text segments. For example, in my patent work it would be good to select relevant passages of patent publications and Wikipedia based on claim or description text.  

### My Motivation

By trying out techniques on a toy problem I can perform some high-level evaluation of what works. Plus it seems fun to match Bible passages to tweets.

### Solution Benefits

The solution can form the basis for text similarity matching algorithms in other projects I am working on.  

### Solution Use

I will write up my results in a blog post report. There will be a separate blog post for each stage in my investigation and a summary page reporting my results.

I could also build a small web application that applies the algorithm to live tweets and other sources of scripture. Who wouldn't want to know which passage of the Qu'ran best matches Donald Trump's latest rant?

---

# How Would I Solve the Problem?

It is worth considering how the problem may be solved manually.

I would start by reading a tweet. I would then manually skim through passages of the Bible until I located one that seemed to be related. If I had a lot of free time I might even rank each passage based on relative relevance (e.g. if there are 1000 passages, each passage would get a unique score between 1 and 1000 - I would definitely lose track after about 3).

Now having to review each passage of the Bible in turn is quite labour intensive. To help I might try to group similar passages together (e.g. these all mention "crows"). I might find that certain portions were more relevant to certain topics (e.g. Genesis to matters of geology and biology).

I might try to summarise each tweet with a set of one or two keywords. I could then use those keywords to do a text search.
