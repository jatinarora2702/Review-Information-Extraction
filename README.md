# Review-Information-Extraction
Code for paper, 'Extracting Entities of Interest from Comparative Product Reviews', CIKM'17

## Basic Idea
While buying a new product, consumers tend to compare products in a similar price range. Not always do the quatitative comparison measures (through product specification), paint the correct picture about the product. Hence, consumers rely on first-hand user experiences by reading through product reviews, on various e-commerce websites, like Amazon, Flipkart, EBay and many more.

Here, we present a framework, which once trained, can identify the main informing entities in a comparitive product review, which I'll explain with an example:

Nikon Coolpix S123 had much better image quality than Nikon XYZ.

1. *Product1*: Nikon Coolpix S123
2. *Product2*: Nikon XYZ
3. *Feature/Aspect*: image quality
4. *Predicate/User-Opinion*: better

## What Next?

Studing and analyzing product reviews, is a popular research area. So many amazing things are getting developed day after day. What I would like to see is:

1. This is not the end of information extraction from product reviews! We are yet to capture trasitivity well. Consider an example:

	Nikon Coolpix has better image quality than Nikon XYZ. But, in terms of durability, I find the latter one better.

2. Only information extraction is not enough. We should put it all up in a knowlege graph and should handle free text user queries. In this way, our search engine can not only present the quantitative comparison (by presenting the product specs), but also, the qualitative comparison (by information extraction from product reviews).

3. The style of presenting views changes a lot with the kind of product we are talking about. So, the way in which I talk about kitchen appliances, is different from clothing, which is again different from electronic items. We felt that the trend of glancing through reviews is majorly popular in the electronics domain, so, we have focussed on that, for now, in our training dataset.

## Acknowledgements

This work would not have been possible without the teachings and guidance from my guides, Prof. Pawan Goyal and Dr. Sayan Pathak. Also, many thanks to my collegue, Sumit Agrawal, for helping me out with this.