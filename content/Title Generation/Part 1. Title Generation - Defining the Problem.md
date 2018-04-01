Title: 1. Title Generation - Defining the Problem
Tags: defining_the_problem
Authors: Ben Hoyle
Summary: This post looks at how we define the problem of title generation.

# 1. Title Generation - Defining the Problem

This post looks at how we define the problem of title generation.  

-----

# What is the problem?

## Problem Introduction

Every patent application has a title. This is based on the subject matter of the application.

The scope of each patent application is defined by the patent claims. These are typically set out at the end of the patent specification. The broadest scope of a patent application is defined by the so-called "independent claims". These are claims that do not reference any other claims. Claim 1 of each patent application is typically the first independent claim.

Our machine learning problem then becomes: can we generate a title given the main patent claim?

## Formalised

A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.

Here:

- T = generate a title.
- E = a corpus of US Patent Publications that include title and claim text.
- P = more difficult - let's look at this in more detail below. 


Classification problems have a simple performance measure: classification accuracy. Here we are trying to generate a string of text, and we have a "ground truth" or original string of text to compare it too. 

One performance measure we could look at is a string difference or distance metric. There are several of these, a common one being the [Levenshtein distance](https://en.m.wikipedia.org/wiki/Levenshtein_distance). One issue with many of these metrics is they heavily penalise different acceptable versions of a word sequence. For example, changing word order may still provide an acceptable output but may have a very high string difference metric value. Also synonyms would likewise lead to a high string distance metric value but may produce an acceptable answer.

For the similar problem of neural machine translation, the [BLEU](https://en.m.wikipedia.org/wiki/BLEU) metric is often used as a performance measure. It could be investigated whether this metric could be used in the present case.

[Perplexity](https://en.m.wikipedia.org/wiki/Perplexity) is another metric that is often used to evaluate text generation models.

The ideal performance measure would be a human score of the generated sentence indicating its suitability. This though is expensive in terms of human effort and is highly impractical for use in model training. There is though the potential to use an actor-critic or [Generative Adversarial Network](https://en.m.wikipedia.org/wiki/Generative_adversarial_network). This architecture uses an additional binary classifier that attempts to predict whether a generated text string came from the set of original text strings or not. Its loss can be used in the loss function of the machine learning model.

## Assumptions

Below are some assumptions that may apply to the problem.

- We assume the content of a patent application may be represented by its first independent claim.
- It is an open question as to whether we could get better results using the full text of the detailed description. However, the first claim of an application gives us a more manageable set of text to work with.
- We assume that there are some underlying patterns to title formation that can be learnt by a machine learning model.


---

## Similar Problems

### Text Summarization

The current problem is similar to the problem of text summarization, i.e. generating a sentence that summarizes a block of larger text.

[This is a useful post from students at Rare Technologies](https://rare-technologies.com/text-summarization-in-python-extractive-vs-abstractive-techniques-revisited/). It looks at the different metrics that may be used and shows some issues that may result from more complex neural models.

Most state of the art models for text summarization appear to be based on a combination of a sequence-to-sequence model (also referred to as  encoder-decoder model) and an attention mechanism.

Jason Brownlee at Machine Learning Mastery has a [post](https://machinelearningmastery.com/encoder-decoder-models-text-summarization-keras/) detailing a simple encoder-decoder model for text summarization in Keras.

Abigail See has a great post [here](http://www.abigailsee.com/2017/04/16/taming-rnns-for-better-summarization.html) that explains a state of the art text summarization system. As well as using a sequence-to-sequence model with attention, Abigail also adds pointers and coverage to solve the problems of factual inaccuracies and repetition.

### Neural Machine Translation

Text summarization is similar to the problem of neural machine translation. There is a larger body of literature on neural machine translation as it is arguably an easier problem to solve: typically you have pairs of input and output sentences in different languages, where there may be similarities in syntax and word order between the input and the output.

Like text summarization, the state of the art models use sequence-to-sequence architectures with attention. As the body of literature is larger for neural machine translation, it may be useful to complement the literature on text summarization.

In terms of practical implementations, there is a detailed tutorial on building a state-of-the-art neural machine translation system in Tensorflow [here](https://www.tensorflow.org/tutorials/seq2seq).

---

# Why does the problem need to be solved?

Generative text models have great promise for automating many tasks. This toy problem will introduce us to many of the aspects (and problems) of more complex text generation systems.

By solving the problem of title generation, we can then move on to bigger problems, such as generating larger blocks of text.

### My Motivation

Text generation is a hard problem in the field of natural language processing. It is an ongoing research area. By addressing a small toy problem for which I have data I can better understand the mechanisms and problems in the field. I can also build a collection of tools and methodologies I can use as a starting point for more complex text generation.

By exploring the data I can also get a feel for the syntactic and semantic structure of patent text.

### Solution Benefits

The solution could be used to automatically generate a patent application title based on the claims.

As the title is short this has limited benefit for reducing drafting time. However, it could be useful for determining keywords or a lower dimensionality claim representation for search. For example, to find prior art, search may be performed in a title space rather than a claim space.

### Solution Use

I will write up my results in a blog post report. There will be a separate blog post for each stage in my investigation and a summary page reporting my results.

The code from the solution may be used in my patentdata project, for example, claim-to-title conversion may be added to patentdata functionality. This could be implemented as a "title()" method on a Claim object. 

---

# How Would I Solve the Problem?

It is worth considering how the problem may be solved manually. 

The guidance on drafting patent specification titles is that the title should not be more limiting that the broadest independent claim. In practical terms, this means that the title should typically only contain words that appear in the main claim. This is a nice limitation for our current problem: the text of the main claim (e.g. claim 1) should ideally provide all the information we need to generate a title.

There are patterns in how titles are drafted. Different patent attorneys follow different conventions. Here are a few widely seen patterns:

* The title is taken from the preamble of a main claim. The preamble is the section of a claim before the phrase "comprising:". For example, a claim may be "A method of painting a house comprising: exploding a paint can." and the preamble is then "a method of painting a house" or even "painting a house". This could be used directly as the title.
* The title features the categories of all the independent claims. For example, if the independent claims relate to a "method", a "system" and a "computer program", the title may be "Method, System and Computer Program [for doing X]". This could cause a problem with our approach as if we use only the text of claim 1, we do not have the categories of the other independent claims. This could be remedied by adding the categories as additional data or using the text of all the independent claims. However, this would be at the cost of increasing the size of our input data. It might be possible for our model to learn the categories that regularly appear and then "guess" these.
* Another pattern is to take the core feature of the independent claim and abstract it. For example, if the new and novel feature of our claim related to a "handle for a spade", where the claim was something like "A spade comprising: a handle with a new widget.", then the title may be "HANDLE FOR A SPADE" or "SPADE HANDLE".
