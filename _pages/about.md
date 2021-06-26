---
permalink: /
title: "Sangho Lee"
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

I'm a MSc student of data science at Seoul National University. Worked in Samsung Electronics forecasting demand of mobile devices and semiconductor, analyzing usage of mobile services (Samsung Pay, Bixby, etc.) for years. Worked as AI engineer in Carrot insurance.

Biography
======
__2021 - present:__ Master student of data science at Seoul National University  
__2019 - 2020:__ AI engineer at Carrot Insurance  
__2013 - 2019:__ Data analyst at Samsung Electronics  
__2006 - 2013:__ BSc at the University of Korea (computer science and mathematics)  


Skills and Certification
======
* Programming: Advanced in Python with Pytorch, Tensorflow, Keras and Scikit-Learn, Intermediate in C++ in Linux/Windows  
* Paper Implementation: Review and Implement ML/Deep learning SOTA papers (github.com/puhuk)  
* Analysis: Statistical understanding of data, Structured/Unstructured data (Image, video, text, etc.) preprocessing, analysis and modeling  
* Languages: Korean (Native), English (Working proficiency)  
* Scholarship: National Science and Engineering Scholarship (2006~ )  
* Others:  
  * Deep learning project mentor (AI bootcamp by Korea Ministry of Employment and Labor) 
  * Sales forecast competition contest top 10% model (Samsung Electronics)
  * Tensorflow Developer Certificate (2020.10.)


Work Experience
------
Work Experience	

Dec. 2020 – Present 	11Street (E-commerce), Seoul, Korea 
∙ Role: Improve search result based on model
∙ Project:
	Improve low quality query result and Implement automated curation
: Statistical analysis of current status and make threshold with low quality query
A/B testing with automated curation and existing curation for analyzing CTR or conversion rate
        Clustering query words based on statistical analysis of user behavior and seller pattern
        Review analysis for extracting products with high retention and response
Nov. 2019 – Dec. 2020 	Carrot Insurance, Seoul, Korea 
∙ Role: Insurance product development based on deep-learning technology
∙ Project:
	Develop crack detection AI model for mobile phone insurance (Dec.2019~)
: AI detection model to detect crack on customer’s mobile device and reject unqualified devices 
Extracts frame from videos and classifies screens, detects crack on the devices with CNN algorithms
(http://www.joseilbo.com/news/htmls/2020/08/20200803403275.html)
	Develop dog breed classification model (Nov.2019~)
: AI module implementation for classifying dog breed in Pet insurance

Jul. 2016 ~ Oct. 2019	Samsung Electronics Mobile Division AI Team 
∙ Role: Statistical analysis of user data and usage prediction (Samsung Pay, Bixby)
∙ Project:
	Bixby user utterance analysis and market response prediction
: Analyze usage data of Bixby from European market with natural language processing algorithms
 	 Prioritize services / languages in each market for increasing retention rate and number of active users
 	 15% increase in Monthly Active Users has achieved
	Service promotion strategy planning
: Market clustering based on usage data and prioritize marketing target and planning for global expansion
 	 Analyze correlation between each factor from usage/market data
	Bixby / Samsung Pay business model setup
: Predict user and amount of each service per countries based on historical data (Device expansion, churn rate, etc.)

Jul. 2013 ~ Jun. 2016	Samsung Electronics Mobile Division Procurement Team
∙ Role: Semiconductor demand forecasting
∙ Project:
	Demand prediction based on product lifecycle, market response, sales history and seasonality
: (KNN clustering for market/semiconductor segmentation, Linear regression for long-term prediction)
Prioritize semiconductors supply chain by with device/semiconductor correlation analysis
Stock amount decreased (over 10%) and Just-In-Time score has increased

![image](https://user-images.githubusercontent.com/2902772/123516332-7d644200-d6d6-11eb-97c0-56031d012262.png)


Create content & metadata
------
For site content, there is one markdown file for each type of content, which are stored in directories like _publications, _talks, _posts, _teaching, or _pages. For example, each talk is a markdown file in the [_talks directory](https://github.com/academicpages/academicpages.github.io/tree/master/_talks). At the top of each markdown file is structured data in YAML about the talk, which the theme will parse to do lots of cool stuff. The same structured data about a talk is used to generate the list of talks on the [Talks page](https://academicpages.github.io/talks), each [individual page](https://academicpages.github.io/talks/2012-03-01-talk-1) for specific talks, the talks section for the [CV page](https://academicpages.github.io/cv), and the [map of places you've given a talk](https://academicpages.github.io/talkmap.html) (if you run this [python file](https://github.com/academicpages/academicpages.github.io/blob/master/talkmap.py) or [Jupyter notebook](https://github.com/academicpages/academicpages.github.io/blob/master/talkmap.ipynb), which creates the HTML for the map based on the contents of the _talks directory).

**Markdown generator**

I have also created [a set of Jupyter notebooks](https://github.com/academicpages/academicpages.github.io/tree/master/markdown_generator
) that converts a CSV containing structured data about talks or presentations into individual markdown files that will be properly formatted for the academicpages template. The sample CSVs in that directory are the ones I used to create my own personal website at stuartgeiger.com. My usual workflow is that I keep a spreadsheet of my publications and talks, then run the code in these notebooks to generate the markdown files, then commit and push them to the GitHub repository.

How to edit your site's GitHub repository
------
Many people use a git client to create files on their local computer and then push them to GitHub's servers. If you are not familiar with git, you can directly edit these configuration and markdown files directly in the github.com interface. Navigate to a file (like [this one](https://github.com/academicpages/academicpages.github.io/blob/master/_talks/2012-03-01-talk-1.md) and click the pencil icon in the top right of the content preview (to the right of the "Raw | Blame | History" buttons). You can delete a file by clicking the trashcan icon to the right of the pencil icon. You can also create new files or upload files by navigating to a directory and clicking the "Create new file" or "Upload files" buttons. 

Example: editing a markdown file for a talk
![Editing a markdown file for a talk](/images/editing-talk.png)

For more info
------
More info about configuring academicpages can be found in [the guide](https://academicpages.github.io/markdown/). The [guides for the Minimal Mistakes theme](https://mmistakes.github.io/minimal-mistakes/docs/configuration/) (which this theme was forked from) might also be helpful.
