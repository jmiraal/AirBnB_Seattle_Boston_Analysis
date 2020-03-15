# Udacity Project: Write a Data Scientist Blog Post

## Nanodegree: Data Scientist

## Title: Airbnb Analysis from Coast to Coast

## Table of Contents

<li><a href="#installation">Installation</a></li>
<li><a href="#project_motivation">Project Motivation</a></li>
<li><a href="#file_descriptions">File Descriptions</a></li>
<li><a href="#results">Results</a></li>
    <ul>
    <li><a href="#perception">1. Perception of the Customers of the Different Types of Hosts </a></li>
    <li><a href="#neighbourhood_dist">2.1 Where are the more expensive neighbourhoods concentrated in each city? </a></li>
    <li><a href="#host_dist">2.2 Which neighbourhoods are prefferred by the hosts with several listings?</a></li>
    <li><a href="#amenities_diff">3. Which are the main differences between Boston and Seattle regarding the amenities announced?</a></li>
    <li><a href="#features_predict">4. Which features are more iportant to predict the price of a listing?</a></li>
    </ul>
<li><a href="#acknowledgements">Licensing, Authors, Acknowledgements</a></li>



 
<a id='installation'></a>
## Installation

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.


<a id='project_motivation'></a>
## Project Motivation

In this project we are going to try to give an answer to the next questions:

1. How is the perception of the customers acording to the reviews received for the different types of hosts? We will make two cassifications for the hosts:

    * Superhost or normal hosts.
    * Hosts with a few lisntings (less than five) and hosts with several listings (more than 5).
    

2. We are going to study the geographical distribution of the listings in the two cities. Where are the more expensive neighbourhoods concentrated in each city? Which neighbourhoods are prefferred by the hosts with several listings.
    
3. Which are the main differences between Boston and Seattle regarding the features published?

4. Which features are more iportant to predict the price of a listing?

<a id='file_descriptions'></a>
## File Descriptions

* **AirBnB_Seattle_Boston_Analysis.ipynb:** A Notebook following the CRISP-DM sequence of steps. 

* **auxiliary_functions.py:** A python file with some fuctions used in the notebook below.

* To do the analysis we have used some datasets downloados from the Kaggle web page:

    * [Boston Datasets](https://www.kaggle.com/airbnb/boston):
    
        - calendar.csv
        - listings.csv
        - reviews.csv
    * [Seattle Datasets](https://www.kaggle.com/airbnb/seattle/data):
    
        - calendar.csv
        - listings.csv
        - reviews.csv 
        
* We include also two maps of Boston and Seattle that can be download from the webpage [OpenStreetMas.org](https://www.openstreetmap.org/#map=12/47.5957/-122.3307) with the coordenates specified in the notebook.

<a id='results'></a>
## Results

The main findings of the code can be found at the post available [here](https://medium.com/@jesus.mira74/airbnb-analysis-from-coast-to-coast-f063c6b28944). This post is very summarized and it doesn't include the predictive part, but I put the results a little more detailed below.

<a id='perception'></a>
### 1. Perception of the Customers of the Different Types of Hosts


* Those hosts with more than five listings usually have worse reviews than the hosts with less listings. This occurs both in Boston and in Seattle. Maybe because of this or because they have to manage a greater number of houses the hosts with more listings have often more availability along the year.


* Analyzing each type of review separately, we have seen that the hosts with more than 5 listings usually have worse scores in all the reviews in Seattle, while they have more similar values in terms of cleanliness in Boston. 


* The superhost program is a recognition that Airbnb grants to some hosts that meet a series of quality conditions, such as high rating in reviews, low cancellation rate, high response rate, etc. According to this is normal that we have seen a clear better general score in the reviews, whereas the difference with other hosts is not so obvious in terms of availability. In Seattle the availability in a year is almost the same for normal hosts and superhosts, while in Boston superhosts have a little more days of availability per year, but the distribution is similar in both groups.


<a id='neighbourhood_dist'></a>
### 2.1 Where are the more expensive neighbourhoods concentrated in each city?

#### Boston:

* In Boston, the more expensive neighbourhoods are concentrated in the center of the city.


* There is also some relation between prices and the cardinal points. The cheaper neighbourhoods tend to be in the South-West of the city, and the prices tend to increase in the North-East.


* The number of listints is bigger in the center of the city also, but there are some cheaper neighbourhoods that have also a considerable number of listints like Jamaica Plain, Allston and Dorchester.

#### Seattle:

* In the case of Seattle, the more expensive neighbourhoods are not so concentrated in the center of the City. Belltown and the Central Business District are centric neighbourhoods with high prices, but there are other expensive neighbourhoods in other regions, normally surrounded by neighbourhoods with medium and low prices. Like Alki, Sunset Hill, Briarcliff, etc


* In general, the cheaper neighbourhoods tend to be in the South or in the North. 


* The neighborhood that stands out for the number of listings is Broadway, but Beltown has an important number of listings too.


<a id='host_dist'></a>
### 2.2 Which neighbourhoods are prefferred by the hosts with several listings?

#### Boston:

* The hosts with several listings preffer usually the more centric and expensive neighbourhoods. In some cases as West End and Chinatown they have around the 60% of the listings.

#### Seattle:

* As in Boston, this type of hosts preffer also the centric neighbourhoods like Beltown, Pike-Market, Central Business Distric, First Hill and Pioneer Square. But there is a clear exception in this case, we can see that the University District in the North has one of the highest rates of hosts with several listings. This is a neighbourhood with low prices, but the presence of an universitary community can have encourage investors to buy houses there with the intention of renting them.

<a id='amenities_diff'></a>
### 3. Which are the main differences between Boston and Seattle regarding some of the features announced?

#### amenities:

We have analysed the percentage of announces that have published each amenity. We have made this calcs for three groups of neighbourghoods (Low, Mediudm and High Priced) in order to be able to compare each type of neighbourhood between the two cities. These are the conclusions:

* In general, the three neighbourhood groups follow the same tendency. Normally, the amenities are more frequent or less frequent in a city in the three categories. The execption is the gyms, that are less frequent in the high priced neighbourhoods in Boston (about a 16%), but equal or more frequent in the medium and low priced neighbourhoods. We can see also that Hot Tub is also around a 16% less announced in Boston in the high priced neighbourhoods, while in the other neighbourhoods the percentage is almost the same. On the other hand Buzzer Wireles Intercom is more published in Boston than in Seattle for the Medium and Low priced neighbourhoods, while it is almos the same for the more expensive neighbourhoods.


* Air Conditioning is again the amenity with the highest increment between Boston to Seattle, but the biggest value occurs in the medium neighbourhoods with a 66% of increment.


* On the other hand, Free Parking on Premises is also the amenity with the highest decrement between Boston to Seattle, but we can see that this difference is specially important in the High Price neighbourhoods with a 52%, while in the small priced neighbourhoods the difference is much smaller with a 28%.


We have also analysed other features like host_verification_type, property_type, room_type and bed_type. We have found the following differences:

#### host_verification_type:
* Most of the listings are verified by email, by the reviews or by phone, both in Boston and in Seattle.


* I don't know the reason, but kba, google, linkedin and facebook are significativaly less used to do verifications in Boston .


* Jumio is a 6.5 % more used in Boston than in Seattle.

#### property_type:

* Apartments are about a 30% more common in Boston than in Seattle and Houses are about a 30% more frequent in Seattle than in Boston. The other property types are very similar.

#### room_type:

* Private Room is a 6% more usual in Boston than in Seattle and Entire home/apt is about a 6% more frequent int Seattle.


<a id='features_predict'></a>
### 4. Which features are more iportant to predict the price of a listing?

We have tested some different options to predict the price. 

* We have applied a Linear Regression Model over the numerical variables and some categorical varibables as: amenities, neighbourhood, host_response_time, property_type, room_type, bed_type and cancellations_policy. The R-squared in the test dataset was: 0.74 in Boston and 0.73 in Seattle. We have seen also that the model have some tendency to overstimate the price.

* We have tried also a Random Forest Model. Applying GridSearchCV to optimize the hyperparameters of the model, we have achieved a R-squared of 0.76 in both cities.

If we pay attention to the Linear Regression Model to try to infere the most important features to predict the price we can extract the following results:

* This sistem gives a lot of relevance to the neighbourhoods. Many of the features that increase or decrease more the prices belong to this category.

* `Doorman` is associated with an increment in the price in Seattle, but the effect is not so relevant in Boston.

* `Other Pets` is associated with an increment of the price in Seattle and a decrement in Boston. On the other hand, `having a washer`is associated with decrement in Seattle and an increment in Boston.

* `Hangers`, `Essentials` and `Cats` are normaly associated with a decrement of the price both in Seattle and in Boston.

* `Smoking Allowed` and `Kitchen` means a decrement of the price more important in Boston than in Seattle.

* `Kitchen` means a decrement in Boston, but is almost neutral in Seattle. 

* `24-Hour Check-In` means a bigger decrement of the price in Seattle than in Boston.


After applying Random Forest we can extract the following conclusions:

* We can see here that, in contrast with linear regression, the neighbourhood is not relevant when we apply Random Forests. In this case, random forests preffers to use the latitude and the longitude to include the localization in the prediction.


* We also see that Private Room is the main feature in Boston while number of Bedrooms is the main Feature in Seattle.


* The position on the map, determined by latitude and longitude, is more important in Boston than in Seattle.

* Other important features are `bathrooms`, `accomodates`, `host listing counts`, etc.

<a id='acknowledgements'></a>
## Licensing, Authors, Acknowledgements

The Airbnb datasets used in this work were downloaded from Kaggle in the links specified at the start of this document. You can see also the licencing in that links (CC0 1.0). 

Regarding to the code of this work, feel free to fork the repository or use it as you want.
