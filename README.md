# WeightingWords

Implementation of different methods for weighing the importance of words for similarity computation in different settings. We used TF-IDF, which indicates how important a term is with respect to the the meaning of a document or collection, and Mutual Information, which indicates the strength of association between two variables such as two words or a word and a type of document. We used the weights in two downstream tasks to examine the performance of the two methods for assigning weights to words.


##Pre-Processing:

We started by taking a training set of five files taken from the web. These files each represent  section or a topic that we will later on use to categorize new data sets. The files we used are:

  * Background.txt
  * Conclusion.txt
  * Results.txt
  * Methods.txt
  * Objective.txt

**In this program, an excerpt is a sentence and a token is a word.**
The first five functions we implemented are concerned with pre-processing, they are the following:

```python
get_all_files(directory)
```
Returns a list of all the file names in a directory.
```python
standardize(rawexcerpt)
```
Tokenizes excerpts and converts them to lowercase.
```python
load_file_excerpts(filepath)
```
Loads all the excerpts within a directory. **An excerpt is a sentence in our case.** So this returns a list of sentences, and each sentence is a list of tokens, which are in turn, words. 
```python
load_directory_excerpts(dirpath)
```
This calls the function before it on all the files within a directory.
```python
flatten(listoflists)
```
This function simply return a flattened list. It takes a list of lists and it returns a simple list.

##TF-IDF

In information retrieval, tf–idf, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. It is often used as a weighting factor in information retrieval, text mining, and user modeling. The tf-idf value increases proportionally to the number of times a word appears in the document, but is offset by the frequency of the word in the corpus, which helps to adjust for the fact that some words appear more frequently in general. Nowadays, tf-idf is one of the most popular term-weighting schemes. For instance, 83% of text-based recommender systems in the domain of digital libraries use tf-idf.

I implemented several helper functions but the main one used is the following, which writes to a new file **"hw1_2-1a.txt"**. I ran this on **sample = "Background.txt", corpus is the set of all five files, k =1000** 
```python
get tfidf topk(sample, corpus, k)
```
**"hw1_2-1a.txt"** will contain the highest weighted words in the sample, along with their corresponsing TF-IDF values. 
Essentially for each word we have used the following formula: 

* IDF (w) = ln(N/DF (w)) where **DF as the count of excerpts in the corpus that contain the word.** and **N to denote the total number of excerpts in the corpus**
* **TF is the raw count of w within in the sample.**

##Mutual Information
```python
get mi topk(sample, corpus, k
```
It takes a sample of text and a larger corpus as input, and outputs a list of (word, weight ) tuples having the highest MI weight in the sample with respect to the corpus in descending weight order. 
My partner then computed p(w) for each token based on the corpus using:

* p(w) = count(w)/sum(count(wj)) 
* MI(sample(i), w) = ln (p(w|sample(i))/p(w))

##Evaluation metric
We used precision and recall to evaluate our algorithms. That is present in the followng two functions: 

* get precision()
* get recall() 

##Labeling new excerpts

The main function implemented here is the one bellow. Essentialy it takes a file of excerpts *(each line is an excerpt)* and it outputs *for each line* the corresponding label; meaning the section that it can be classified as. 
```python
label sents(excerptfile, outputfile)
```
We used several helper functions but the idea is summarized in the following steps:

* Obtain our vocabulary of words which is the most important words in the entire corpus based on their tf-idf values (we ran  get tf_idf_topk(corpus, corpus, 1000)
* Make the vocanulary into a vector and obtain the values of each of the five files we have in function of this vector.
* Vectorize our excerpt (testing data) in terms of the vocabulary
* Compute cosine similarity between each section vector and the excerpt vector and take the closest one.

I tested the accuracy of the output against excepts that were already classified and I obtain the following result.

Accuracy of labeling:
![Alt text](/s1.png?raw=true "Screenshot1")
