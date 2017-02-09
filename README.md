# WeightingWords

Implementation of different methods for weighing the importance of words for similarity computation in different settings. We used TF-IDF, which indicates how important a term is with respect to the the meaning of a document or collection, and Mutual Information, which indicates the strength of association between two variables such as two words or a word and a type of document. We used the weights in two downstream tasks to examine the performance of the two methods for assigning weights to words.


##Pre-Processing:

We started by taking a training set of five files taken from the web. These files each represent  section or a topic that we will later on use to categorize new data sets. The files we used are:
*Background.txt
*Conclusion.txt
*Results.txt
*Methods.txt
*Objective.txt

The first five functions we implemented are concerned with pre-processing, they are the following:

```python
get_all_files(directory)
standardize(rawexcerpt)
load_file_excerpts(filepath):
load_directory_excerpts(dirpath)
flatten(listoflists):
```


Accuracy of labeling:
![Alt text](/s1.png?raw=true "Screenshot1")
