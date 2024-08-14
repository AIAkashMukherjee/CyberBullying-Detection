# CyberBullying Detection


## About Dataset

This dataset contains annotated social media comments specifically curated for the analysis and detection of cyberbullying. The data is labeled across multiple categories including race, religion, gender, sexual orientation, and miscellaneous attributes. Each comment has been carefully reviewed and annotated by multiple annotators to ensure accuracy and reliability.

## Original Dataset in hateXplain was in JSON Format

### Sample Entry ->

{

"1179055004553900032_twitter": {

"post_id": "1179055004553900032_twitter",

"annotators": [
{"label": "normal", "annotator_id": 1, "target": ["None"]},
{"label": "normal", "annotator_id": 2, "target": ["None"]},
{"label": "normal", "annotator_id": 3, "target": ["None"]}
],

"rationales": [],

"post_tokens": ["i", "dont", "think", "im", "getting", "my", "baby", "them", "white", "9", "he", "has", "two", "white",
"j", "and", "nikes", "not", "even", "touched"]

}

## This Dataset is a Scraped and Well-Processed Version of the above data, specially designed for Meta Data Extraction

### hateXplain.csv

**post_id** : Unique id for each post

**annotators** : The list of annotations from each annotator

**annotators[label]** : The label assigned by the annotator to this post. Possible values: [Hatespeech, Offensive, Normal]

**annotators[annotator_id]** : The unique Id assigned to each annotator

**annotators[target]** : A list of target community present in the post

 **rationales** : A list of rationales selected by annotators. Each rationale represents a list with values 0 or 1. A value of 1 means that the token is part of the rationale selected by the annotator. To get the particular token, we can use the same index position in "post_tokens"

 **post_tokens** : The list of tokens representing the post which was annotated

### final_hateXplain.csv

#### Different Label Categories and Their Respective Values

 **Race** : no_race, african, arab, asian, caucasian, hispanic, indian, indigenous

 **Religion** : nonreligious, buddhism, christian, hindu, islam, jewish

 **Gender** : no_gender, men, women

 **Sexual Orientation** : no_orientation, asexual, bisexual, heterosexual, homosexual

 **Miscellaneous** : none, disability, economic, minority, other, refugee

### ACKNOWLEDGEMENT ->

I am here citing the original work and the creators

[@inproceedings](https://www.kaggle.com/inproceedings){mathew2021hatexplain,
title={HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection},
author={Mathew, Binny and Saha, Punyajoy and Yimam, Seid Muhie and Biemann, Chris and Goyal, Pawan and Mukherjee, Animesh},
booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
volume={35},
number={17},
pages={14867--14875},
year={2021}
}

#### Model used Random Forest, accuracy Score: 71 %
