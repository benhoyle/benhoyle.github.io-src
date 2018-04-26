Title: 1. Learning Grammars - Defining the Problem
Tags: defining_the_problem
Authors: Ben Hoyle
Summary: This post looks at how we can improve text generation by using learnt grammars.

# 1. Learning Generation - Defining the Problem

This post looks at how we can improve text generation by using learnt grammars.

-----

# What is the problem?

## Problem Introduction

Recurrent neural networks have been used widely to attempt to solve the problem of text generation. Andrej Karpathy helped to popularise the use of recurrent neural networks for text generation via his blog post: [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). In Andrej's post, he used a character-level recurrent neural network to produce Shakespeare-esque portions of text. Yoav Goldberg, my favourite natural language processing skeptic, wrote a follow-up post explaining that this unreasonableness was not necessarily a sole feature of recurrent neural networks: even simple counting functions (~30 lines of Python) can produce something similar. The main surprise is that language can has structure that is fairly easy to learn, up to a point...

Now we come to the caveat: the counting and recurrent neural networks appear impressive when generating short snippets. However, they tend to fall down on closer inspection and more real-world implementations. Here are some of the problems we find:

* Rambling sentences;
* Simple grammar mistakes; and
* Nonsense.

Rambling sentences arise because the models tend to operate on the *n* previous tokens (whether characters or words). There tends to be little or no long-range structure or consistency. Also models tend to fall upon, and get stuck within, loops of cliche, common phrases that occur often (and so have high probabilities) but provide little meaning.

The models are trained by just throwing millions of tokens at a wall and seeing what weights stick. The models learn grammars by learning patterns in sequences of tokens. However, language is complex and every sentence has multiple patterns that are operating together. The models tend to produce coherent sequences across 10-20 character tokens, or 1-5 word tokens, but struggle when combining these into more complex and consistent structures.

Finally, many models generate text that looks at first glance to be English, but when read makes absolutely no sense. In particular, the deeper structure of language seems to be missing: that sentences are about things; that consecutive sentences need to be consistent; and that things may be described at different levels of detail using different vocabulary.

Our problem is thus: how can we improve these generative text models?
 
In particular, how can we produce text that feels more like real-world text.

## Grammars

This project is based on the premise that we can improve generative text models through the use of learnt grammars.

A [grammar](https://en.wikipedia.org/wiki/Grammar) may be defined at a simple level as a set of structural rules governing the composition of clauses, phrases, and words in any given natural language.

Many people have a negative experience of grammars. The word "grammar" sums up memories of school lessons learning seemingly endless rules from musty text books. Certain pedants treat the rules of grammar as rules akin to the laws of physics - fixed and to be militantly enforced. However, grammar is simiply what we call the patterns of structure that occur in language. These patterns can and do change over time, but stabilise in the short term based on social agreement (with some help from the pedants).

Machine learning algorithms seek to learn patterns in data. As grammar may be seen as the patterns of structure in language, can we use machine learning algorithms to learn these patterns?

### Grammars vs RNNs

When we start to look at grammars, we soon see a problem. A defining feature of many grammars seems to be [recursion](https://en.wikipedia.org/wiki/Recursive_grammar): a rule may be applied to itself over and over again. This may be contrasted with the natural sequential processing of recurrent neural networks.

For example, recursion may be seen in the the sentence: "Jim thought that Alice said that Berty ate grass". In this phrase, we have nested subject-verb-object sets: e.g. "Berty ate grass" > 1, "Alice said that \[1\]" > 2, "Jim thought that \[2\]", where the same rule appears to be applied at different levels.

Recurrent neural networks cope fairly well with simple sentences such as 1 above. However, they appear to stuggle to represent the higher level nesting, despite the rule being the same and relatively simple. This may be because recurrent neural networks are working from token to token, storing a state that represents the context. It may be that multi-layer architectures can learn such rules, but the greater the abstraction, the more difficult it is to arrive at the structure (in the form of weight values) from low level training data. This lack of stability may be seen in the "catastrophic forgetting" of neural networks, where previous rules become lost as more training data is supplied.

There have been attempts to introduce alternative Recursive Neural Networks. A great bit of work in this area has been performed by Richard Socher of Stanford (see his [thesis on Recursive Deep Learning for Natural Language Processing](https://nlp.stanford.edu/~socherr/thesis.pdf)). However, for various reasons (including confusion with recurrent neural networks and difficulty training) they have not found widespread adoptance.

### Competing Grammars

It is worth remembering at this stage that there is no definitive theoretical grammar framework. There are several different forms of grammar, including:

* "Generative" or "Phrase-based" grammars that apply rules to generate phrases based on parts of speech;
* "Dependency" grammars that look at the relationships between pairs of words in a sentence; and
* "Functional" grammars that look at how words are used in communication, e.g. to indicate agents, subjects and objects.

Each framework looks at the rules of language use from a different angle, and each says something useful about language use. Each, though, is necessarily incomplete.

It is also worth remembering that the twin paths of linguistics and computing often run parallel to each other and have their own cultures. For example, linguists are often based within social science departments, and historically have defined themselves based on "Capital Letter Theories" as opposed to large-scale computation and experimentation. Computer scientists, however, have traditionally built models that represent small-scale patterns in language but that ignore the messier complexities of actual language use. 

Luckily, we are not bound by any particular discipline, and can pick the best from both worlds with an eye to utility in generation.

### Data Available
*** To be finalised ***
What data do we have available?

Spacy parse.

Iterate through sentences.

Build the graph from the dependency parse.

When a token has no children: this is a terminal node. We can store the POS > token. Mapping. For tokens with children, we can look at left and right dependencies. This sets the direction? If we have a single dependency chain > this can become a single mapping rule?

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
