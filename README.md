# GeoKnowr

A lightweight GeoGuessr AI.

Created by Sho Kiami and Zach Chapman.

<br>

## Table of Contents

1. [Abstract](https://github.com/shokiami/GeoKnowr/blob/main/README.md#abstract)
2. [Problem Statement](https://github.com/shokiami/GeoKnowr/blob/main/README.md#problem-statement)
3. [Related Work](https://github.com/shokiami/GeoKnowr/blob/main/README.md#related-work)
4. [Methodology](https://github.com/shokiami/GeoKnowr/blob/main/README.md#methodology)
5. [Results](https://github.com/shokiami/GeoKnowr/blob/main/README.md#results)
6. [Examples](https://github.com/shokiami/GeoKnowr/blob/main/README.md#examples)
7. [Demo](https://github.com/shokiami/GeoKnowr/blob/main/README.md#demo)
8. [Video](https://github.com/shokiami/GeoKnowr/blob/main/README.md#video)
9. [Looking Forward](https://github.com/shokiami/GeoKnowr/blob/main/README.md#looking-forward)

<br>

## Abstract

[GeoGuessr](https://www.geoguessr.com/) is a popular web game where users are thrown into random locations around the world in Google Street View and are challenged to place a marker on the world map to guess where they are in the world (the closer you guess, the more points you get).

There are many variables that go into making a good guess: climate, architecture, street signs, vegetation, vibes, and more. However, what if we could save people the brainpower required to make an educated guess? That's where __GeoKnowr__ comes in!

GeoKnowr is a neural network that has been trained on tens of thousands of street view images from around the world, and is able to guess with a respectable amount of accuracy the location of any street view image thrown its way.

<br>

## Problem Statement

In GeoGuessr, a user could be tasked to locate an image like the following:
<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206887420-2371c244-5248-4723-b6db-2bccaf79a4ff.png" width="98%"/>
</p>
<br>

Our goal was to use computer vision and deep learning to create a GeoGuessr AI that would be able to reliably guess the location of where such images were taken. Furthermore, GeoGuessr has several different modes, one of which is NMPZ (no moving-panning-zooming) which is notoriously the most difficult and thus the one we wanted to tackle.

There are three major aspects to this probelem: data collection, training, and testing. For us, we started with neither data nor a model, and thus we had to build up the entire pipeline from scratch.

<br>

## Related Work

At this point, we would like to acknowledge all third-party technologies/tools that inspired or helped us along the way:
- GeoGuessr (obviously).
- Google Maps and Google Street View for data collection and testing (all of our street view imagery came from Google's APIs).
- PyTorch pretrained ResNet-18 (more on this later) which we used for transfer learning ([link](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) to ResNet paper). This model was trained on a subset of the ImageNet dataset, consisting of around 1.3 million images for standard image classification.

<br>

## Methodology

### __Data Collection__

Our data collection could be broken up into the following steps:
1. Choose a random (latitude, longitude) coordinate.
2. Use Google's API's to see if any Google Street View locations exist within a 10km search radius.
3. If so, grab the metadata for that location and scrape the corresponding street view image at a random heading.
4. Repeat steps 1-3 until we gather enough data.

We started off collecting high resolution images (1920x1080), but later realized that we would just scale them down for our model anyway, making the extra time/space not worth it. In the end, using this method, we downloaded a total of 32,000 images of (480x360) resolution from around the world. We would have loved to have more data, but chose against it due to deadline and space constraints.

<br>

### __Training__

We decided to frame this as a regression problem, with the goal of minimizing surface distance around the unit sphere because this is ultimately the criterion we are trying to minimize when playing GeoGuessr. For our model, we initially started off with a custom convolutional neural network, however, our results were not great.

One of the issues we noticed was that our model would consistently guess Greenland. This made sense, as Greenland is a relatively central location to most of our data (most of the world is in the northern hemisphere and Google Street View has an overrepresenetation of European data), and so by using regression, our model simply learned to average guess.

To illustrate part of the issue with regression, below is an image of Google's street view coverage. Notice how not all of the Earth has been documented.
<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206890186-f5f8e7cf-3607-42e9-9652-c890a838bc5a.png" width="98%"/>
</p>

In order to combat this issue, we tried several other models including another custom network, this time with residual connections, in an attempt to improve the results--to no avail.

The first major breakthrough came when we realized that we could reframe the problem as classification by dividing up the globe into numerous regions. The model would then classify an image into one of these regions and then guess the center of the region. This forced our model to stop average guessing and guess more directly, as nearby regions are equally penalized as regions on the opposite side of the world. Another motivation behind this switch was the recognition that as humans, we also play GeoGuessr by region guessing--mentally dividng up the world into discrete sections.

First, we tried dividing up the world into a grid of equal sized sectors. However, the issue with this was that the majority of the sectors had very few examples in our data (e.g. over the water or in an area with low Google Street View coverage). As a result, similarly to regression, our model would simply learn to continuously predict the majority class.

To combat this, we came up with a clever solution of using clustering algorithms to perform the class divisions for us, hopefully leading to more equal sized classes (with less sparsity in our data). After trying several different clustering algorithms and numbers of classes, we found that the Gaussian mixture model worked the best and 21 classes was the sweet spot where less classes led to regions which were too large and more classes led to too little examples per class. Here are the final clusters we ended up using:
<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206887434-f334025f-8a0b-4601-be02-f6cec9b9c7d7.png" width="98%"/>
</p>

Each colored region is a seperate class that our model tries to categorize images into. Note how our classification map mostly lines up with Google Street View's covereage.

Unfortunately, the model was still predicting the majority class, albeit sometimes throwing in one or two other classes. We concluded that this was an issue with the size of our data set; we did not have enough data to adequately train a deep neural network from scratch.

The second major breakthrough came when we recognized the possibility of leveraging transfer learning. We tried transfer learning on several different models and found ResNet-18 to yield the best performance considering its ease of training. This made a _huge_ difference; our model no longer had to learn how to extract core features from the images such as specific edges and shapes. Instead, it would gain access to the features provided by pretrained ResNet-18 and could focus on learning the relationship between those features and our classes.

After an abundance of experimentation, we settled on the following hyperparameters:
- Classes: 21
- Epochs: 15
- Batch Size: 32
- Learning Rates: 0.001, 0.0005, 0.0001
- Weight Decay: 0.0001
- Optimizer: Adam

Note: we utilized learning rate annealing where we trained for 5 epochs at each learning rate.

<br>

### __Evaluation__

Our main metrics by which we evaluated our model throughout the training process was cross entropy loss (standard to classification) and model accuracy. For example, at one point we noticed that our model was heavily overfitting to our training data which prompted us to add weight decay and tune our other hyperparameters.

However, in the end, the metric we cared most about was the real-world distance between the model's guesses and the ground truth. To measure this we developed a test script which evaluates the model on the test set (see results section) and visualizes several random examples (see examples section). Note that this is a better assessment of our model's performance over relying on test accuracy--while a guess may be labelled in the incorrect class, in actuality, it could be very close in terms of real-world distance.

Furthermore, we also considered how reasonable the model's guesses were in general (e.g. confusing Canada with Greenland is much more reasonable than confusing Canada with Nigeria). This was much more hand-wavy, however, and thus we did not have a great way to measure this quantitatively.

<br>

## Results

Here are our resuls after training for 15 epochs (~5 hours):
- Final Train Loss: 2.016952
- Final Test Loss: 2.153938
- Final Train Accuracy: 36.56%
- Final Test Accuracy: 32.75%
<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206887495-7fdc886c-3ea6-43c3-b58c-076f9fd598d3.png" width="49%"/>
  <img src="https://user-images.githubusercontent.com/43970567/206887497-eba00fa3-0787-4e25-ab8d-d591a42ecd3c.png" width="49%"/>
</p>

As mentioned previously, the metric we care most about is the distribution over the real-world distances from the model's guesses to the ground truth.
<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206887447-de077199-f887-4577-ba6a-a2ccf88fddb1.png" width="98%"/>
</p>

We are very happy with these results! We speculate our model will be able to beat the average Joe at GeoGuessr (average Joe, not our amazing professor Joe).

<br>

## Examples

Below are 10 example images and corresponding guesses from our model. Note that the red marker represents the ground truth and the grey marker represents the model's guess.

Eurajoki, Finland: 46.15km away <br>
GT: (61.24206624516949, 21.49451874391871) <br>
Guess: (60.866010738653614, 21.13157246788378) <br>
<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206888820-66a1873e-eb41-47b2-a55d-12b8ad1bc2ca.png" width="49%"/>
  <img src="https://user-images.githubusercontent.com/43970567/206888821-432c50a8-5752-4149-9a73-291c34bee6a0.png" width="49%"/>
</p>

Cedar Pocket, Australia: 766.68km away <br>
GT: (-26.2016867183339, 152.7428709700448) <br>
Guess: (-31.684634839922268, 147.96179641452864) <br>
<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206888822-7e10529d-0f19-440b-93c4-117418db1ff9.png" width="49%"/>
  <img src="https://user-images.githubusercontent.com/43970567/206888825-80097726-4992-4144-a64e-752164b66f6b.png" width="49%"/>
</p>

Ōdai, Japan: 554.07km away <br>
GT: (34.29128958773475, 136.2255376621636) <br>
Guess: (39.220061536212654, 137.1401260865559) <br>
<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206888827-d75b62cf-1e3b-43f7-b773-0ec345854d13.png" width="49%"/>
  <img src="https://user-images.githubusercontent.com/43970567/206888829-88e9ed80-ecca-46ac-a80d-7dc21479d5c1.png" width="49%"/>
</p>

Pervomaiskii, Russia: 750.37km away <br>
GT: (54.67348511884279, 54.76858543638147) <br>
Guess: (57.3687329171972, 65.85774195636928) <br>
<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206888834-f3933270-84da-4f8d-96e2-6fad9f1fc678.png" width="49%"/>
  <img src="https://user-images.githubusercontent.com/43970567/206888835-7e059711-ea12-4a2d-8436-b1f2daed0ba8.png" width="49%"/>
</p>

Clavering Øer, Greenland: 3001.91km away <br>
GT: (74.36102804428536, -20.31095398871028) <br>
Guess: (62.88085149124142, -94.28640809358343) <br>
<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206888850-1f2d64da-8ca6-414f-b38b-8fd78b10f156.png" width="49%"/>
  <img src="https://user-images.githubusercontent.com/43970567/206888855-04dfde27-6af6-4c67-a056-3c4e251e42a4.png" width="49%"/>
</p>

Nuenen, Netherlands: 728.77km away <br>
GT: (51.47712383096619, 5.568049157904403) <br>
Guess: (46.52608328096249, -0.99022351617955) <br>
<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206888857-fee0d68e-57ec-4ab1-b613-0ce6975f396e.png" width="49%"/>
  <img src="https://user-images.githubusercontent.com/43970567/206888858-7fb9fbdd-0664-4585-a96b-f21104e58b52.png" width="49%"/>
</p>

Tanjung Mulia, Indonesia: 778.63km away <br>
GT: (2.111117753198836, 100.2296566933782) <br>
Guess: (9.086239024023117, 99.60871719478148) <br>
<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206888859-7527d46e-5ad7-47fe-8b07-0d549cbdfdda.png" width="49%"/>
  <img src="https://user-images.githubusercontent.com/43970567/206888860-1544f91b-e64a-4096-aa65-bc163e8258a4.png" width="49%"/>
</p>

Takper, Nigeria: 860.73km away <br>
GT: (7.055152796196378, 8.483934407321177) <br>
Guess: (9.65765106509962, 1.1147835438224805) <br>
<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206888862-be5e87c1-19c0-4c96-977c-3a59578e2f98.png" width="49%"/>
  <img src="https://user-images.githubusercontent.com/43970567/206888866-258517b1-a0b1-4d30-ab0c-31e4f4d58e8a.png" width="49%"/>
</p>

Colonia Río Escondido, México: 26.80km away <br>
GT: (28.57000772521148, -100.6163712162074) <br>
Guess: (28.465027052700034, -100.36940739027031) <br>
<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206888869-f568abc2-26aa-4a3c-89fa-42716d495c6a.png" width="49%"/>
  <img src="https://user-images.githubusercontent.com/43970567/206888871-f9ef5239-664d-48d4-b768-b4dd7383d72f.png" width="49%"/>
</p>

Chipaya, Bolivia: 1133.61km away <br>
GT: (-19.00475769083379, -68.10597816923791) <br>
Guess: (-8.813989360376178, -68.40059804082549) <br>
<p align="middle">
  <img src="https://user-images.githubusercontent.com/43970567/206888873-7c88c588-5faf-4e36-86c4-6fb6fcabd952.png" width="49%"/>
  <img src="https://user-images.githubusercontent.com/43970567/206888877-a88f17f1-fd9b-4f24-880c-64309aeca9af.png" width="49%"/>
</p>

<br>

## Demo

Try it yourself! Follow these steps:
1. Clone the repo:
    ```
    git clone git@github.com:shokiami/GeoKnowr.git
    ```
2. Add images into the folder `demo_in`. We have included some example images for you. :)
3. Run:
    ```
    python3 demo.py
    ```
4. Navigate to the outputted url's to see the model's guesses in Google Maps.

<br>

## Video

Watch our video presentation:
[![video](https://user-images.githubusercontent.com/43970567/206905313-86b6c4ff-08f6-4919-8f57-fdd6355acb68.png)](https://youtu.be/F2mRi_Vl4hg)

<br>

## Looking Forward

We are very pleased with the results that we have seen, yet there are still improvements that can be made. For starters, we only have 32,000 street view data points, which is very small considering the amount of Google Street View coverage and how difficult the problem space is. With more data, not only will the network naturally perform better after training, but we would be able to increase the number of classes, allowing finer-grained guessing which would in turn increase model performance. 

Another avenue we want to explore in the future is returning to using regression. Especially if we were to gather more data, we hope that our implementation of techniques such as transfer learning would help our model better learn the correlations in our data instead of average guessing.
