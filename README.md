# Review Information Extraction

Code for paper, 'Extracting Entities of Interest from Comparative Product Reviews', CIKM'17

**Paper:** [Extracting Entities of Interest from Comparative Product Reviews](http://cse.iitkgp.ac.in/~pawang/papers/cikm17.pdf)

The dataset and model image of the results presented in the paper can be downloaded from this link, [Model and Dataset](https://zenodo.org/record/1415481#.W5pjkBwScnQ).

## Basic Idea

While buying a new product, consumers tend to compare products in a similar price range. Not always do the quatitative comparison measures (through product specification), paint the correct picture about the product. Hence, consumers rely on first-hand user experiences by reading through product reviews, on various e-commerce websites, like Amazon, Flipkart, EBay and many more.

Here, we present a framework, which once trained, can identify the main informing entities in a comparitive product review, which I'll explain with an example:

Nikon Coolpix S123 had much better image quality than Nikon XYZ.

1. **Product1**: Nikon Coolpix S123
2. **Product2**: Nikon XYZ
3. **Feature/Aspect**: image quality
4. **Predicate/User-Opinion**: better

## What Next?

Studing and analyzing product reviews, is a popular research area. So many amazing things are getting developed day after day. What I would like to see is:

1. This is not the end of information extraction from product reviews! We are yet to capture trasitivity well. Consider an example:

	Nikon Coolpix has better image quality than Nikon XYZ. But, in terms of durability, I find the latter one better.

2. Only information extraction is not enough. We should put it all up in a knowlege graph and should handle free text user queries. In this way, our search engine can not only present the quantitative comparison (by presenting the product specs), but also, the qualitative comparison (by information extraction from product reviews).

3. The style of presenting views changes a lot with the kind of product we are talking about. So, the way in which I talk about kitchen appliances, is different from clothing, which is again different from electronic items. We felt that the trend of glancing through reviews is majorly popular in the electronics domain, so, we have focussed on that, for now, in our training dataset.

## Acknowledgements

This work would not have been possible without the guidance and support from my guides, [Prof. Pawan Goyal](http://cse.iitkgp.ac.in/~pawang/) and [Dr. Sayan Pathak](https://www.linkedin.com/in/sayan-pathak-19abb42/). Also, many thanks to my collegue, Sumit Agrawal, for his contributions to the project.

## For Citation

[Extracting Entities of Interest from Comparative Product Reviews](https://dl.acm.org/citation.cfm?id=3133141)

### ACM Reference

Jatin Arora, Sumit Agrawal, Pawan Goyal, and Sayan Pathak. 2017. Extracting Entities of Interest from Comparative Product Reviews. In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management (CIKM '17). ACM, New York, NY, USA, 1975-1978. DOI: https://doi.org/10.1145/3132847.3133141 

### BibTeX
```
@inproceedings{Arora:2017:EEI:3132847.3133141,
 author = {Arora, Jatin and Agrawal, Sumit and Goyal, Pawan and Pathak, Sayan},
 title = {Extracting Entities of Interest from Comparative Product Reviews},
 booktitle = {Proceedings of the 2017 ACM on Conference on Information and Knowledge Management},
 series = {CIKM '17},
 year = {2017},
 isbn = {978-1-4503-4918-5},
 location = {Singapore, Singapore},
 pages = {1975--1978},
 numpages = {4},
 url = {http://doi.acm.org/10.1145/3132847.3133141},
 doi = {10.1145/3132847.3133141},
 acmid = {3133141},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {comparison mining, deep learning, opinion extraction},
} 
```
