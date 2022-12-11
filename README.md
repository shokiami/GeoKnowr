# GeoKnowr

A GeoGuessr AI, created by Sho Kiami and Zach Chapman

Abstract - in a paragraph or two, summarize the project
-------------------------------------

[GeoGuessr](https://www.geoguessr.com/) is a popular website that throws users into a random Google street view location, challenging them to accuratly guess where in the world they are. They could be thrown into Japan, Paris, the jungles of Africa, or--heaven forbid--Ohio. There are many variables that go into making a good prediction, along with the need for a well-rounded grasp of the world. However, what if we could save people the brainpower required to make an educated guess? That's where __GeoKnowr__ comes in, cutting down the time to make accurate guesses, and in the process freeing up precious minutes that can instead be spent leaving Ohio.

GeoKnowr is a neural network that has been trained on tens of thousands of street view images from around the world, and is able to guess with a respectable amount of accuracy the location of any street view image thrown its way.

<br>

Problem statement - what are you trying to solve/do
---------------------------------------

We wanted to create a neural network that, given an image from Google street view, is able to beat out a regular person at guessing where in the world the picture was taken. We say regular person as there are some who have dedicated hundreds of hours to learning the ins and outs of GeogGuessr and thus have nearly perfect accuracy, down to a few hundred km. They would be very hard to beat with just a few weeks of development.

There are multiple modes of play within GeoGuessr, and the one we want our model to excel at is the no-movement mode. Instead of being able to wander around the street view to find clues, this is a hard-core mode where you only get a single frame to guess off of. And yes, the GeoGuessr pros still have nearly perfect accuracy even under such a tough restriction, so we're not trying to compete with them.

There are three major facets to such a probelem: data collection, training, and testing. No matter how good a model can be, it is entirely useless without a sufficient amount of data to train on. And without testing, who knows if a model is useable or not. For us, we started with neither data or a model, and thus had
to design methods to complete all three.

<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206887420-2371c244-5248-4723-b6db-2bccaf79a4ff.png" width="1000"/>
</p>
Above is a screenshot of us playing GeoGuessr. Note in the bottom left a world map that allows us to guess where we think we are. Japan seems probable based on what we saw, but it sure would be nice for a neural network to take away thinking on our end.

<br><br>

Related work - what papers/ideas inspired you, what datasets did you use, etc
---------------------------------------------

After deciding to switch over to transfer learning, we utilized quite heavily a pre-trained ResNet18 model ([Link](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) to ResNet paper). This model was trained on a sebset of the ImageNet dataset, consisting of around 1.3 million images.

<br>

Methodology - what is your approach/solution/what did you do?
---------------------------------------------------

Data Collection
---------------------------------------------------

To generate enough data for our model to work with, we deciding to go with a scaping method. We continuosly call the Google street view website at random coordinates until our get request returns 'OK'. Once this occurs, we open up a playwright webkit browser at that location, scrub away unneeded UI elements, and save a screenshot.

We started off collecting high res images (1920x1080), but later realized that we would just scale them down for our model anyway, making the extra time not worth it.

Using this method we downloaded a total of 32,000 images of 360p quality from around the world. If we had more time additional data would of course be welcome, but that comes with a high time price, both on the scraping and on the model training.

Training
---------------------------------------------------

We decided to frame this as a regression problem, with the goal of minimizing arc distance around the unit sphere (surface distance). We thought that this would produce guesses with the lowest distance from the truth. For our model, we initially started off with a custom convolutional nn (neural network) model, with which we had decent success but nothing spectacular. We later switched over to another custom network, this time residual, in an attempt to improve the results. While this did help a little, results remained in the same ballpark.

One of the issues we noticed was how our model would consistently guess Greenland. This made sense, as Greenland is a relatively central location to most of our data (Streetview has an overrepresenetation of European data), and because we were using regression. We realized that to fix this issue we could reframe the problem as classification, with different regions of the world making up our classes. This forced our model to stop making "median guesses", and start guessing more directly.

<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206890186-f5f8e7cf-3607-42e9-9652-c890a838bc5a.png" width="1000"/>
</p>

While the swap from regression to classification improved accuracy and made the predictions more human-like, we were still not getting great results. We knew that the size of our dataset was quite limited for training from scratch, and looked for ways we could squeeze out better results. With this in mind, we swapped from custom networks to a pre-trained ResNet18 model using transfer learning. This made a _huge_ difference, as all of a sudden our model no longer had to waste time learning edges and trees, and could instead get right to learning location differences.

<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206887434-f334025f-8a0b-4601-be02-f6cec9b9c7d7.png" width="1000"/>
</p>
Above is our classification map. Each colored region is a seperate class that our model tries to accurately predict. Using classification regions produced better results than regression, as the classes help the model learn what regions of the world are similar. We also have enough classes that distance off is typically pretty close. 

Testing
---------------------------------------------------

While we did switch from regression to classification, we still wanted to be able to see how far off the model's guesses were, and not just if they were in the right class or not. So, in addition to testing loss and accuracy, we added testing for distance, taking the latitude and longitude of both the guess and the ground truth, and then conducting the proper calculations.

<br>

Experiments/evaluation - how are you evaluating your results
------------------------------------------------------

When we get the results of our testing we are looking for a few trends. Of course we want loss to be as low as possible and accuracy to be as high as possible, but we also want our guesses to either be close to the truth, or if not, a reasonable guess. One of our examples further down the page show a guess of Canada when the actual is Greenland. While the distance of this is quite high, it's acutally not a bad result as this could be an easy mistake for a human to make as well. Parts of Greenland look just like parts of Canada. We found that our best performing model not only has good accuracy, but makes very reasonable guesses even when incorrect. On this same point, the actual performance of our model is better than what the accuracy may show, as being close but in a nieghboring class is counted as incorrect.

We also found from our results that we were overfitting our training data, which we helped reduce the severity of by adding weight decay. We used our results to great effect in finding hyperparameters that worked the best for what we were looking for.

<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206887495-7fdc886c-3ea6-43c3-b58c-076f9fd598d3.png" width="500"/>
  <img src="https://user-images.githubusercontent.com/43970567/206887497-eba00fa3-0787-4e25-ab8d-d591a42ecd3c.png" width="500"/>
</p>

<br>

Results - How well did you do
--------------------------------------------------

Once we figured out how to get the most out of our training, we were able to hit a training loss of #, a test loss of #, and training accuracy of #, and a test accuracy of #.

We are very happy with these results, as going into this project our goal was to beat an average person. With the guesses we have seen our model make, we would absolutely destroy the average Joe (average Joe, not our amazing professor Joe).

<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206887447-de077199-f887-4577-ba6a-a2ccf88fddb1.png" width="1000"/>
</p>

<br>

Examples - images/text/live demo, anything to show off your work (note, demos get some extra credit in the rubric)
--------------------------------------------------------

Eurajoki, Finland: 46.15km away <br>
GT: (61.24206624516949, 21.49451874391871) <br>
Guess: (60.866010738653614, 21.13157246788378) <br>
<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206888820-66a1873e-eb41-47b2-a55d-12b8ad1bc2ca.png" width="500"/>
  <img src="https://user-images.githubusercontent.com/43970567/206888821-432c50a8-5752-4149-9a73-291c34bee6a0.png" width="500"/>
</p>

Cedar Pocket, Australia: 766.68km away <br>
GT: (-26.2016867183339, 152.7428709700448) <br>
Guess: (-31.684634839922268, 147.96179641452864) <br>
<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206888822-7e10529d-0f19-440b-93c4-117418db1ff9.png" width="500"/>
  <img src="https://user-images.githubusercontent.com/43970567/206888825-80097726-4992-4144-a64e-752164b66f6b.png" width="500"/>
</p>

Ōdai, Japan: 554.07km away <br>
GT: (34.29128958773475, 136.2255376621636) <br>
Guess: (39.220061536212654, 137.1401260865559) <br>
<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206888827-d75b62cf-1e3b-43f7-b773-0ec345854d13.png" width="500"/>
  <img src="https://user-images.githubusercontent.com/43970567/206888829-88e9ed80-ecca-46ac-a80d-7dc21479d5c1.png" width="500"/>
</p>

Pervomaiskii, Russia: 750.37km away <br>
GT: (54.67348511884279, 54.76858543638147) <br>
Guess: (57.3687329171972, 65.85774195636928) <br>
<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206888834-f3933270-84da-4f8d-96e2-6fad9f1fc678.png" width="500"/>
  <img src="https://user-images.githubusercontent.com/43970567/206888835-7e059711-ea12-4a2d-8436-b1f2daed0ba8.png" width="500"/>
</p>

Clavering Øer, Greenland: 3001.91km away <br>
GT: (74.36102804428536, -20.31095398871028) <br>
Guess: (62.88085149124142, -94.28640809358343) <br>
<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206888850-1f2d64da-8ca6-414f-b38b-8fd78b10f156.png" width="500"/>
  <img src="https://user-images.githubusercontent.com/43970567/206888855-04dfde27-6af6-4c67-a056-3c4e251e42a4.png" width="500"/>
</p>

Nuenen, Netherlands: 728.77km away <br>
GT: (51.47712383096619, 5.568049157904403) <br>
Guess: (46.52608328096249, -0.99022351617955) <br>
<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206888857-fee0d68e-57ec-4ab1-b613-0ce6975f396e.png" width="500"/>
  <img src="https://user-images.githubusercontent.com/43970567/206888858-7fb9fbdd-0664-4585-a96b-f21104e58b52.png" width="500"/>
</p>

Tanjung Mulia, Indonesia: 778.63km away <br>
GT: (2.111117753198836, 100.2296566933782) <br>
Guess: (9.086239024023117, 99.60871719478148) <br>
<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206888859-7527d46e-5ad7-47fe-8b07-0d549cbdfdda.png" width="500"/>
  <img src="https://user-images.githubusercontent.com/43970567/206888860-1544f91b-e64a-4096-aa65-bc163e8258a4.png" width="500"/>
</p>

Takper, Nigeria: 860.73km away <br>
GT: (7.055152796196378, 8.483934407321177) <br>
Guess: (9.65765106509962, 1.1147835438224805) <br>
<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206888862-be5e87c1-19c0-4c96-977c-3a59578e2f98.png" width="500"/>
  <img src="https://user-images.githubusercontent.com/43970567/206888866-258517b1-a0b1-4d30-ab0c-31e4f4d58e8a.png" width="500"/>
</p>

Colonia Río Escondido, Mexico: 26.80km away <br>
GT: (28.57000772521148, -100.6163712162074) <br>
Guess: (28.465027052700034, -100.36940739027031) <br>
<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206888869-f568abc2-26aa-4a3c-89fa-42716d495c6a.png" width="500"/>
  <img src="https://user-images.githubusercontent.com/43970567/206888871-f9ef5239-664d-48d4-b768-b4dd7383d72f.png" width="500"/>
</p>

Chipaya, Bolivia: 1133.61km away <br>
GT: (-19.00475769083379, -68.10597816923791) <br>
Guess: (-8.813989360376178, -68.40059804082549) <br>
<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206888873-7c88c588-5faf-4e36-86c4-6fb6fcabd952.png" width="500"/>
  <img src="https://user-images.githubusercontent.com/43970567/206888877-a88f17f1-fd9b-4f24-880c-64309aeca9af.png" width="500"/>
</p>

<br>

Video - a 2-3 minute long video where you explain your project and the above information
--------------------------------------------------------------


<br>

Future potential
---------------------------------------------------

We are very pleased with the results that we have seen, and yet there is still improvements that can be made. For starters, we only have 32,000 street view data points, which compared to the size of the earth is pretty small (though it still performed well despite that). An easy way to get better performance without too much effort is just to scrape more data. If we have a lot more data, not only will the network naturally perform better after training, but we would be able to increase the number of calssifications, allowing finer grained guessing.
