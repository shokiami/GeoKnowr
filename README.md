# GeoKnowr

A GeoGuessr AI.

Abstract - in a paragraph or two, summarize the project
-------------------------------------

[GeoGuessr](https://www.geoguessr.com/) is a popular website that throws users into a random Google street view location, challenging them to accuratly guess where in the world they are. They could be thrown into Japan, Paris, the jungles of Africa, or--heaven forbid--Ohio. There are many variables that go into making a good prediction, along with the need for a well-rounded grasp of the world. However, what if we could save people the brainpower required to make an educated guess? That's where __GeoKnowr__ comes in, cutting down the time to make accurate guesses, and in the process freeing up precious minutes that can instead be spent leaving Ohio.

GeoKnowr is a neural network that has been trained on tens of thousands of street view images from around the world, and is able to guess with a respectable amount of accuracy the location of any street view image thrown its way.

<br>

Problem statement - what are you trying to solve/do
---------------------------------------

We wanted to create a neural network that, given an image from Google street view, is able to beat out a regular person at guessing where in the world the picture was taken. We say regular person as there are some who have dedicated hundreds of hours to learning the ins and outs of GeogGuessr and thus have nearly perfect accuracy, down to a few hundred km. They would be very hard to beat with just a few weeks of development.

There are three major facets to such a probelem: data collection, training, and testing. No matter how good a model can be, it is entirely useless without a sufficient amount of data to train on. And without testing, who knows if a model is useable or not. For us, we started with neither data or a model, and thus had
to design methods to complete all 3.

<br>

Related work - what papers/ideas inspired you, what datasets did you use, etc
---------------------------------------------

After deciding to switch over to transfer learning, we utilized quite heavily a pre-trained ResNet18 model ([Link](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) to ResNet paper). This model was trained on a sebset of the ImageNet dataset, consisting of around 1.3 million images.

<br>

Methodology - what is your approach/solution/what did you do?
---------------------------------------------------

__Data:__ <br>
To generate enough data for our model to work with, we deciding to go with a scaping method. We continuosly call the Google street view website at random coordinates until our get request returns 'OK'. Once this occurs, we open up a playwright webkit browser at that location, scrub away unneeded UI elements, and save a screenshot.

We started off collecting high res images (1920x1080), but later realized that we would just scale them down for our model anyway, making the extra time not worth it.

Using this method we downloaded a total of 32,000 images from around the world. If we had more time additional data would of course be welcome, but that comes with a high time price, both on the scraping and on the model training.

__Training:__ <br>
We decided to frame this as a regression problem, with the goal of minimizing arc distance around the unit sphere (surface distance). We thought that this would produce guesses with the lowest distance from the truth. For our model, we initially started off with a custom convolutional nn (neural network) model, with which we had decent success but nothing spectacular. We later switched over to another custom network, this time residual, in an attempt to improve the results. While this did help a little, results remained in the same ballpark.

One of the issues we noticed was how our model would consistently guess Greenland. This made sense, as Greenland is a relatively central location to most of our data (Streetview has an overrepresenetation of European data), and because we were using regression. We realized that to fix this issue we could reframe the problem as classification, with different regions of the world making up our classes. This forced our model to stop making "median guesses", and start guessing more directly.

While the swap from regression to classification improved accuracy and made the predictions more human-like, we were still not getting great results. We knew that the size of our dataset was quite limited for training from scratch, and looked for ways we could squeeze out better results. With this in mind, we swapped from custom networks to a pre-trained ResNet18 model using transfer learning. This made a _huge_ difference, as all of a sudden our model no longer had to waste time learning edges and trees, and could instead get right to learning location differences.

__Testing:__ <br>
While we did switch from regression to classification, we still wanted to be able to see how far off the model's guesses were, and not just if they were in the right class or not. So, in addition to testing loss and accuracy, we added testing for distance, taking the latitude and longitude of both the guess and the ground truth, and then conducting the proper calculations.

<br>

Experiments/evaluation - how are you evaluating your results
------------------------------------------------------

<br>

Results - How well did you do
--------------------------------------------------

Once we figured out how to get the most out of our training, we were able to hit a training loss of #, a test loss of #, and training accuracy of #, and a test accuracy of #.

We are very happy with these results, as going into this project our goal was to beat an average person. With the guesses we have seen our model make, we would absolutely destroy the average Jo (average Jo, not our amazing professor Jo).

<br>

Examples - images/text/live demo, anything to show off your work (note, demos get some extra credit in the rubric)
--------------------------------------------------------

<br>

Video - a 2-3 minute long video where you explain your project and the above information
--------------------------------------------------------------
