Title: 1. Classifying Claims - Defining the Problem
Tags: defining_the_problem
Authors: Ben Hoyle
Summary: This post looks at how we define the problem of claim classification.

# 1. Classifying Claims - Defining the Problem

This post looks at how we define the problem of claim classification.  

-----

# What is the problem?


## Problem Introduction

The problem is that each patent application needs to be classified into one of a set of standardised groups.

These groups are based on different categories of technical subject matter. For example, there may be a group based on electronics or chemistry. There are also classification hierarchies with different layers of classification.

There are several different standardised classification schemes. A widely-used scheme is the [International Patent Classification (IPC)](http://www.wipo.int/classifications/ipc/en/) system. Classifications in the IPC are split into four levels: Section (a letter); Class (two digits); Subclass (a letter); and Group. The Group consists of one to three digits followed by a slash followed by at least two digits. The first set of one to three digits represent a main group and the digits after the slash represent a sub-group.

As an example, H01S 3/00 is a classification for the "Lasers" main group, in Section "H" for "Electricity", Class "01" for "Basic Electric Elements", and Subclass "1" for "Devices Using Stimulated Emission". The newer Cooperative Patent Classification (CPC) is a more detailed version of the IPC.

The scope of each patent application is defined by the patent claims. These are typically set out at the end of the patent specification. The broadest scope of a patent application is defined by the so-called "independent claims". These are claims that do not reference any other claims. Claim 1 of each patent application is typically the first independent claim.

A more precise problem is how to assign an IPC/CPC Section to a first claim of a patent application.

## Formalised

A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.

Here:

- T = assign an IPC/CPC Section to claim 1 of a patent application.
- E = a corpus of US Patent Publications that include claim text and assigned classifications.
- P = classification accuracy, the number of claims correctly assigned a Section out of all claims, considered as a percentage.

## Assumptions

Below are some assumptions that may apply to the problem.

- We assume the content of a patent application may be represented by its first independent claim.
    - It is an open question as to whether we could get better results using the full text of the detailed description.
    - However, the first claim of an application gives us a more manageable set of text to work with.
-  We assume that each patent application has a single classification.
    - In practice a patent application may have several classifications.
    - This may influence classification accuracy for borderline cases, which may fall into multiple groups.
    - We may thus need to create and display a confusion matrix.
- We assume the words of the claim contain enough information to separably assign a Section.
    - This may not always apply for particularly broad claims that use abstract language - for these cases the full text of the application may be better.
- Often the abstract of a patent application is based on the main independent claim. The abstract is used to summarise the subject matter of the application. Hence, limiting to main claims appears a valid restriction.

## Similar Problems

There are eight main Sections in the IPC:

- A Human Necessities
- B Performing Operations; Transporting
- C Chemistry; Metallurgy
- D Textiles; Paper
- E Fixed Constructions
- F Mechanical Engineering; Lighting; Heating; Weapons; Blasting Engines or Pumps
- G Physics
- H Electricity

Our problem then appears to be a multi-class classification problem with 8 different classes. The input for our problem is a paragraph of text. Hence, the problem appears similar to many other text classification problems, e.g. where categories need to be assigned to blocks of text. For example, similar problems include labelling emails or classifying citation abstracts.

---

# Why does the problem need to be solved?

Assigning a patent classification helps allocate patent work to different teams and departments. The chosen group may also be used to select different models and automated processing. For example, chemistry patent cases may need to be managed and processed differently to computing patent cases.

### My Motivation

The problem seems a good way to learn about applying supervised classification techniques to patent data. By tackling the problem I hope to improve my skills in applying classification algorithms to text. For example, processing claims and converting text into a numeric vector is likely to be a common task in patent processing.

Also, I'd like to practice applying recent neural network and deep learning techniques and a project methodology to help structure my analysis.

By working through and commenting on each step, the resulting write-up should also be useful as an educational piece.

### Solution Benefits

If I am able to get good classification results I can use the model as a way to classify claims during drafting so as to retrieve an appropriate subset of prior publications. For example, I may be able to parse a newly drafted claim and then know to be searching in Section x for prior publications.

The result may also provide a stepping stone to more detailed classifications moving down the hierarchies of the patent classifications.

A solution will also allow me to demonstrate my skills in applying machine learning techniques to patent data. Certain processing steps may be developed from the project as standard functions.

### Solution Use

I will write up my results in a blog post report that can also be exported to PDF. There will be a separate blog post for each stage in my investigation and a summary page that links to the detailed work.

The code from the solution may be used in my patentdata project, for example, claim-to-vector conversion may be added to patentdata functionality. This could be implemented as a "classify()" method on a Claim object.

---

# How Would I Solve the Problem?

It is worth considering how the problem may be solved manually.

First I would obtain the titles associated with each Section. I would then read the text of the claim and determine what title was the most appropriate. This will in most cases be a gut feeling.

Reflecting on why I made a particular choice, reasons may be that particular words in the claim suggested an association with a particular Section. Certain words would generally not be informative, such as "a method" or "comprising". These words would be the most frequently used words across the complete set of claims. Certain Sections may have associated claim formats, for example, claims relating to chemistry may feature chemical symbols and biotech inventions may use Latin names. Rare words may be informative in certain cases, e.g. certain acronyms.

To solve this problem manually I would need to collect a print out of the claim text. For example, I could print one claim per page.  I could then write a Section on each page.
