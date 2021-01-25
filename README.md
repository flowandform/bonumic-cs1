# <div align="center"> **Named Entity Recognition for real-estate flyers**

&nbsp;

## Meeting Industry’s Requirement
<div align="justify">
Commercial real estate is a broker-based industry that suffers from a lack of a centralized source of information. A single, searchable database/inventory with frequently enough updated information does not exist. This presents a great challenge for brokers and makes them often rely on networking and  a variety of third party databases containing incomplete and outdated information.
It comes as no surprise then that  there are a number of companies which aim to monetize this situation by attempting to produce and deliver organized real estate inventories. However, this is not an easy task since the inventory is being updated on a daily basis and there is a large amount of commercial real estate properties out there to keep track on.
In order to acquire the data these companies sometimes contact brokers for info or even visit physical sites, but even more so they collect information from flyers which usually contain all relevant information regarding the real estate offering. In fact, digital flyers are the most commonly used method of sharing offering information and basically all  commercial real estate offerings are accompanied with a flyer. In other words, real estate flyers are one of the most important sources of information for commercial real estate. However, the problem is that collecting information from them is very troublesome. It is a very expensive and labour intensive process since it is a combination of manual information extraction and manual data entry to the database.

## The Flyers

Real estate flyers are free form digital marketing materials usually provided as a PDF datatype. Since they are a multi-media document their structure can vary a lot - they can differ a lot in visual aspects,  can contain floor plans, satellite maps,  can consist of multiple pages, and can contain different levels of information regarding the address, broker contacts, size, offering type and much more.

<div align="center">
<img src="https://raw.githubusercontent.com/flowandform/bonumic-cs1/main/src/img/Clipboard01.jpg" width="700">
<font size="2">  Example of a flyer
</div> </font>

&nbsp;


## The problem at hand

Our goal was to explore the possibility of creating a product that could continuously update a database of offerings with important information extracted from the real estate flyers. In order to accomplish this one must overcome many different obstacles along the way, but generally the overall procedure can be considered as a two step process - i) Optical Character Recognition (OCR) and ii) Named Entity Recognition (NER).

OCR is a process of converting (detecting and extracting) textual information from images into an editable and searchable output format.  Images can be  in various image, video formats and PDF documents, while output files can be formats such as .txt, Word or even Excel documents. Nowadays there are plenty of software that can do this in an automatic fashion. Some of them are commercial such as Abby and Adobe Reader, some are free & open source such as Tesseract, Kraken, Kalamari and some come as a paid cloud service such as Google Cloud Vision. The algorithms powering these software are sophisticated and usually include some form of Machine or Deep Learning, but due to their inherent complexity they do not fall under the scope of this blog. Also, if interested in comparing various OCR software, there already exists a number of blogs on this topic out there such as [here](https://source.opennews.org/articles/so-many-ocr-options/) and [here](https://dida.do/blog/how-to-choose-the-best-ocr-tool-for-your-project).

Named Entity Recognition is a branch of Information Extraction that focuses on automatic extraction of  structured information from unstructured or semi-structured text documents. More precisely, NER deals with extracting very specific and user predefined information which are then called named entities. In the context of Machine Learning, NER falls under the category of supervised learning which assumes training a Machine/Deep Learning model on a labeled dataset. Since NER and supervised learning are very wide areas that include many algorithms which were developed over a period of many years, this blog will not go deeper into explaining these topics. Therefore, for a reader interested in reading the basics about these subjects two great blog posts are recommended ([I](https://towardsdatascience.com/named-entity-recognition-and-classification-with-scikit-learn-f05372f07ba2) and [II](https://towardsdatascience.com/supervised-vs-unsupervised-learning-14f68e32ea8d)).

Moreover, OCR and NER are fields of research which belong to two very big application domains of AI & Machine Learning, which are computer vision and natural language processing (NLP) respectively. These fields deal with analysing digital images/videos and human language data (text and voice).

## General workflow

<div align="center">
<img src="https://raw.githubusercontent.com/flowandform/bonumic-cs1/main/src/img/Clipboard02.jpg" width="700">

<font size="2">  Workflow diagram
</div> </font>


Whether you are only curious about general OCR and NER procedures or your interest lies specifically with real estate flyers, or if perhaps your plan is to try something similar, create a Minimum Viable Product (MVP) or a full software product, the general workflow presented in this blog and visible on the picture should provide you with a decent overview of the overall process of extracting key information from digital real estate flyers.

## Gathering data
As your first goal, you should aim to create a small and relatively simple model which will serve both as a proof of concept of the project but also as a baseline for comparison with more complex models you create down the road. Same as any other Machine Learning project, workflow in this one also starts with gathering data (flyers) for model training.

First and foremost, a minimum number of flyers used for initial training should be defined. This number will depend on the overall complexity of the problem - i) on your modelling target group, e.g. is your goal to create a model for extracting information from specific flyer provider or do you wish to create a model that can be applied on a wide range of flyers, ii) on intrinsic flyer complexity and quality, e.g.lot of background images with various and/or dark colours, iii) between-flyers variance, iv) sufficient frequency of named entities you wish to extract across the flyers, v) entity-level variance, and many more factors.

A very important thing of course, is to make sure that the data which is being collected for training is a statistically representative sample of the flyers which you expect to encounter in the production. It would be disappointing if you were to spend weeks creating a highly sophisticated model only to realize that the production data, the one you wish to apply your model on, is significantly different than the data your model was trained on.

In our current workflow, we have decided that approximately 700 flyers would be sufficient to create a Proof of Concept.

## Data cleaning and preprocessing
One of the necessary initial steps in any ML project is data cleaning. In this context, this would mean making sure that each and every one of the flyers going through the OCR process are the types of flyers you really want to model and that the files are of good quality. There is a possibility to encounter corrupt files, flyers that are not entirely real estate related, low quality flyers and etc. There is also a possibility of encountering duplicate files, both in name and in content. Also, files could come in different data formats and therefore this is something that should also be standardized before moving forward. These are all the steps we have done either visually before applying any code, or at the beginning of our Machine Learning pipeline which you can find at our Github [repository](https://github.com/flowandform/bonumic-cs1/tree/main/src/scripts).

While you are sure to have a certain number of miscellaneous pre and post processing procedures in a machine learning project like this one, there are a number of preprocessing steps which are common in the field of computer vision and NLP. Image preprocessing techniques usually try to apply some type of transformation on an image with the goal of making it as easy as possible for the OCR engine to differentiate the text from the background. These include experimenting with image contrast, sharpness, scale, removing noise, image binarization or grayscale conversion, etc. Reality is that OCR software can work really great on your “typical page of text" with perfect resolution, black characters and white background, but it might also struggle in other circumstances such as the real estate flyers. More on improving the OCR process with preprocessing techniques can be found on various blogs such as [I](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html) and [II](https://docparser.com/blog/improve-ocr-accuracy/).

NLP preprocessing steps usually consist of some combination of applying the following processing steps: tokenization, removing stopwords, punctuations and special characters, noise removal, converting to lowercase, stemming and lemmatization, part of speech tagging, BIO tagging, noun phrase chunking, etc. Purpose of these steps is to remove characters which usually aren't helpful for NLP, standardize variations of words or to enrich text to help the model to learn better. In our workflow we have experimented mostly with conversion to grayscale and binarization and concluded that grayscale versions of our flyers gave the best results in OCR.

<div align="center">
<img src="https://raw.githubusercontent.com/flowandform/bonumic-cs1/main/src/img/Clipboard03.jpg" width="700">

<font size="2">  Grayscale version of the real estate flyer
</div> </font>

Regarding NLP preprocessing, which came after applying OCR and before doing the NER modelling, we have applied tokenization only. Reason for this is the nature of the texts in the real estate flyers - unlike typical texts in articles, books and so on, which are usually completely sequential and where sentences are related with each other, a large amount of text in the flyers is scattered and unrelated with each other. Of course, in the following period after the proof of concept is approved, some of the preprocessing and enrichment steps will be experimented with as part of the pipeline.

## OCR and postprocessing
As our tool of choice we have chosen Tesseract, an open source OCR engine developed by Google. Among other things, Google uses this engine for text detection on mobile devices, in video, and in Gmail image spam detection. More precisely, we have used pytesseract, since our programming language for this project was Python. If a reader requires any help with setting up the tool, there are plenty of useful video and blog instructions out there, such as this 4 minute youtube [video](https://www.youtube.com/watch?v=4DrCIVS5U3Y).

Our input dataset consisted of approximately 700 flyers in .pdf, .jpeg and .png format. After performing the data cleaning and preprocessing steps mentioned earlier, we have experimented with the parameters that the Tesseract has as options, primarily Page Segmentation Modes and Engine Mode. Page Segmentation Mode defines how the engine will split the image in box(es) of text and words. By default, Tesseract expects a “normal” page of text, which is definitely not a case with real estate flyers. In our case, the most optimal parameter was --psm 11, which assumes that there is no particular number, shape, alignment or size of the text chunk(s) in the document. In other words, it coerces the engine to find as much text as possible in no particular order. As for the Engine mode, the default option was the optimal one, which uses both the legacy model and a Long short-term memory (LSTM) neural network, a type of a recurrent neural network (RNN).

More details about Tesseract parametrization can be found [here](https://ai-facets.org/tesseract-ocr-best-practices/), and on the official user [manual](https://tesseract-ocr.github.io/tessdoc/).

<div align="center">
<img src="https://raw.githubusercontent.com/flowandform/bonumic-cs1/main/src/img/Clipboard04.jpg" width="500">

<font size="2">  Output of applying Tesseract on a pdf flyer
</div> </font>

While various preprocessing procedures mentioned earlier are usually satisfactory methods of improving the OCR success rate for most users, sometimes it might be worthwhile to explore the option of training the OCR model yourself using the Tesseract. Choice is then between training the model from scratch using one of the engines/algorithms Tesseract has to offer, or to train on top of a pretrained model provided also by the Tesseract. Details about training can be found on the official manual or on this excellent blog [here](https://www.endpoint.com/blog/2018/07/09/training-tesseract-models-from-scratch).

Some other options of acquiring better OCR results would be to try to use a different OCR software or even to combine results of multiple OCR systems.
Since OCR results are rarely perfect even when applying OCR on a very simple page of text, there is often a possibility of improving the result by applying some form of editing to the text. This is possible if there are some patterns in errors the OCR engine is making on your documents, which is then possible to programmatically modify to remove the errors. For example, there are some characters that OCR sometimes gets confused about, such as detecting character ‘S’ as ‘5’, ‘l’ as ‘1’ and etc. Solutions to this and similar errors mostly come down to inserting, deleting or substituting a single or multiple characters inside words, word splitting, word merging and etc. There are sometimes more complex scenarios but these are usually problem specific and can require some custom magic as a solution.

As part of our pipeline, there was one additional step of postprocessing required due to our choice of tools for annotation, Doccano, which will be explained slightly below. As a requirement for this tool, the text output of our OCR-ed flyers needed to be condensed - extra white spaces and all line breaks needed to be deleted. Therefore, our final output of the OCR process needed to be one single .txt file, where each line of text was actually the whole text extracted from a particular real estate flyer.

## Doccano annotations

As mentioned earlier, NER is a supervised Machine Learning problem. This means that, when providing training data to the model we also need to provide labels for each data point. In other words, data needs to be annotated. In the context of NER, this means that for each line of the text we need to in some way specify all the entities and positions of the entities that we want our model to learn on. In a sense, by doing this we are “telling” the model what we want him to learn to recognize in the text and hope that, given a sufficient number of training data and a proper setup/parametrization of the model, he will be able to recognize these entities on the flyers he had never “seen”. In our project, we have decided to label and train model for learning the following 7 entities: Email, GPE(countries, cities, states), Person, Pricing, Size, Street address, Zip code. Entity Email was chosen as a sort of a sanity check for later purposes, since extracting emails is something that is very easy to do by applying regular expressions, i.e. by finding words which contain the symbol ‘@’.

In theory, this is something that can be done manually by editing the OCR text outputs in a standardized way required by the type of the algorithm you plan to use for modelling. However, annotation process is very prone to human errors, and is also a very tiresome procedure especially when doing named entity recognition. For this purposes there are several tools out there which make this process much easier such as Doccano, Prodigy, etc.

We have used an open source annotation tool, Doccano, which functions as a server on which you upload your text data, annotate it and export it in a form suitable for further modelling. The data can be imported and exported in one of the several possible formats, for us it was easiest to use plain .txt as import format, and .jsonl as export format. The great thing about tools such as Doccano is that it simplifies the process of annotations for users. It enables the annotators to easily navigate between texts and tag the entities by dragging the mouse, using shortcuts and distinct colours for entities.


<div align="center">
<img src="https://raw.githubusercontent.com/flowandform/bonumic-cs1/main/src/img/Clipboard05.jpg" width="600">

<font size="2">  Example of using Doccano annotation tool for NER
</div> </font>


## Training a NER model
At this point, real estate flyers are OCR-ed and converted to text, post processed, annotated and are ready to be used for training a NER model. Spacy, an industrial-strength natural language processing library, was our tool of choice for the NER modelling. Spacy is easy to set up in python and makes it possible to experiment with NER within an hour. [Here](https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da) you can find a nice blog which demonstrates basic usage of the Spacy NER model.

A pretty neat thing about Spacy is that it comes with three pretrained multitask convolutional neural networks, which allow for recognition of 18 different entities (list of [entities](https://spacy.io/api/annotation#section-named-entities)). These models differ in size, complexity and accuracy of results. They were trained onto a OntoNotes 5 [corpus](https://catalog.ldc.upenn.edu/LDC2013T19), a cummulative publication of 2,9 million words from various sources (telephone conversations, newswire, newsgroups, broadcast news, weblogs).

Even though some of the entities present in the spacy NER model are the ones we are interested in such as person and GPE, most of the entities are not there. Also, even the ones who are did not provide satisfying results when applying raw pretrained NER on our annotated data. Of course, because of these reasons we needed to train the NER model ourselves with the inclusion of the new entities which are of interest to us. Therefore, our goal was to train on top of the pre-trained neural network, to utilize both the existing “knowledge” and to expand it to learn on our own entities. Details of training this algorithm are out of the scope of this blog post, but can be found on a well documented code on our GitHub repo and on the Spacy manual webpage.


## Results
Since we have used an already trained model for the OCR part of the Machine Learning pipeline and did not have a training and test dataset per se, we had to take a manual approach for analysing the accuracy of applying Tesseract on our data. For this purpose, we have chosen a subset of representative flyers and written down all the entities appearing in the original real estate images/pdf files. Then, we have done the same for the text outputs of the OCR procedure, and compared the differences. Specifically, we have counted the number of successes, partial successes and failures for each entity. OCR was considered a success when the converted entity had a 100% with the one in the original file. You can see the results in the table below.

<div align="center">
<img src="https://raw.githubusercontent.com/flowandform/bonumic-cs1/main/src/img/Clipboard06.jpg" width="600"></div>

&nbsp;


For some entities, such as Size, Tesseract did a decent job in finding and converting the text from images. Results for Street address on the other hand are not that satisfactory, whereas for Pricing the conclusion can not be made since this entity did not appear in sufficient number of flyers.  Overall, the results are moderate but this is not something that we find negative. Reason for this is that, since this was an MVP, we did not experiment nearly as much as we could with all the possible preprocessing steps and other OCR tools that we could. Moreover, one very probable option which we did not experiment yet with for improving the results significantly is to train an OCR model on top of the pretrained one, which is a functionality that Spacy also offers. One of additional reasons why we think that OCR results can be greatly improved is that the quality of real estate flyers is always high which is often not the case when doing OCR in some other applications such as OCR-ing old documents, handwritten texts, signs on photographs and so on.

NER modelling results were much easily obtained since we had a designated test set with corresponding labels for comparison with the ones that model would output. Examples of some of the results and the overall result are shown in the tables below.

<div align="center">
<img src="https://raw.githubusercontent.com/flowandform/bonumic-cs1/main/src/img/Clipboard06.jpg" width="600">

<font size="2">  Example of model outputs for 4 different entities
</div> </font>

<div align="center">
<img src="https://raw.githubusercontent.com/flowandform/bonumic-cs1/main/src/img/Clipboard07.jpg" width="600">

<font size="2">  Overall model result
</div> </font>

Results for all entities, except for Pricing which as mentioned above did not have a sufficient number of appearances in the flyers, are very satisfying. Average F1 score without the Pricing is 0.92. If a reader wishes to learn more about F1 score here is a good [explanation]( https://en.wikipedia.org/wiki/F1_score), but to keep it simple one can consider the F1 score as something that is similar (colloquially speaking) to accuracy. 1 would be the perfect soore, 0 would mean nothing was recognized successfully.

What makes these results even more satisfying is the fact that, similarly as for the OCR part of the Pipeline, here we also did not go into depths with experimenting with various steps that could make the results even better. This means that there is also a lot of room for improvement here too. However, it is also necessary to mention that there is probably some slight level of overfitting present in the model due to the nature of our training and testing data. This is because variety for some entities in our flyers was not that high. For example, there are some cases where same people's names (in our case real estate brokers) can be found both in our training in testing data. Therefore, there is a possibility that the model has only remembered some of the entities, rather than actually learning to recognize them in a general way.

## Next steps

Based on the combined results of the OCR and NER we have concluded that proof of concept was created. Very specific information can be extracted from real estate flyers with satisfying accuracy using Machine Learning techniques, and thus there is a great potential for this approach to replace the great amount of manual labor that humans in real estate need to do.
First step in doing so would be to train a model on a much larger set of data than we had, and then to research and experiment with the possible techniques of improvement mentioned in the post.
On top of that, there are some other very interesting methods of improving the overall results that could be explored for our problem too. Usually, NER approaches are based on textual information only. However, since real estate flyers are visually rich documents, a lot of important entity information on them can be conveyed by visual elements such as colour, entity position and size. On this topic there are some excellent articles out there, such as the Combining Visual and Textual Features for Information Extraction from Online Flyers by Apostolova E. & Tomuro N.
With the state of the art model developed, the model is going to be deployed to production. However, this is a topic for another blog post, as is the process of significantly improving the model. Take care!
