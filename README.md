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


Accuracy of labeling:
![Alt text](/s1.png?raw=true "Screenshot1")
