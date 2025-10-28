# Hurricane Center Computer Vision Project
A tool to mark the "best track center" and intensity prediction of a hurricane from satellite images using various satellite products alongside in situ measurements.

## Part 1 - Conceptual design

### Problem description

In general, tropical cyclones are storms that involve rotating low-pressure weather systems without any "fronts" (air masses with different densities). They are named and classified in their intensities by the maximum sustained windspeeds and the location of their formation. Hurricanes, for example, are ones that form in the Atlantic basin, eastern or central North Pacific Ocean and that have a sustained windspeed of 74mph or greater [National Ocean Surface](https://oceanservice.noaa.gov/facts/hurricane.html).  With the availability of satellite imaging, experts can use the Dvorak method to classify the intensity, growth, and decay of tropical cyclones using a variety of visual classification techniques [Tropical cyclone intensity analysis using satellite data](https://repository.library.noaa.gov/view/noaa/19322) often at a temporal resolution of 2 minutes or less [[Do et al](doi.org/10.1038/s41598-025-12733-w)].    

### Conceptual questions and learning
What will I need to learn to solve the problem?
I will need to take the satellite image and have the algorithm learn where the centers are placed.  I think this should be the most straightforward task. Then, using the aforementioned Dvorak method, the shape might also be classified which gives more insight into the intensity and growth or decay of the tropical cyclone. Therefore, this could be similar to "face recognition problems" where specific shapes and features are emblematic of specific identities.
Do you need to detect the object (i.e., tell where it is) in the frame?
This is a good question that I do not know the answer to yet exactly - depending on where the hurricane is in the image, it may be clear that it is in a central location (some satellite images are already centered at the hurricane much like a headshot of a person). Others may be in a bigger view and so having a sense of where it is would be important.
If so, is there anything specific about the object you are detecting you would like to use (e.g., color, shape, “key points” that are very specific to that object, etc.)? 
I will be using data that has images in a variety of bands or wavelengths of light. This may help increase or decrease the reliability of the image predictions, or it may be extraneous, unnecessary information like the color channels of face images as we talked about in class. 
What kind of features you think would be needed to calculate, and which image properties your solution should be agnostic to?
I will also attempt to incorporate the background, in-situ data that can give some baseline truth that may help extract more information from the images.

### Data sources

#### Satellite

The data will be sourced from satellite images from NOAA GOES-R Series Advanced Baseline Imager (ABI) Level 1b Radiances which have 16 channels of near-infrared and infrared bands.
[NOAA GOES](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C01501). 

#### Dropsondes

Dropsondes are small, cylindrical measurement tools that are released from airplanes that fly over and in the hurricane to collect a variety of measurements including windspeed, wind direction, relative humidity, and pressure among other things. These are often used "operationally" to classify and categorize the various aspects of a hurricane.  
[Dropsonde data](https://www.aoml.noaa.gov/hrd/data_sub/dropsonde.html)

#### Sea Surface Temperature Data

Sea surface temperature, along with the related ocean heat content, are thought to be an important aspect of the fueling of hurricanes. There are a variety of products that are used to extract this information, including drifters, gliders, XBTs, and buoys [NOAA Hurricane data](https://www.aoml.noaa.gov/data-products/#hurricanedata) 


## Part 2 - Data acquisition and preparation
Prepare a short description of a database that you acquired (no need to upload an actual database into GitHub or Google Drive). Push the report (under "Part 2" section in your readme.md) to your semester project repo by the deadline. Include these elements into your report:

### Source (download link and associated paper(s) offering the dataset(s))
The tropical cyclone (named typhoon because of the location in the Northwester Pacific Basin, hereto referred as TC). The paper is located at [Multi-heat keypoint incorporation in deep learning model to tropical cyclone centering and intensity classifying from geostationary satellite images](https://www.nature.com/articles/s41598-025-12733-w) after making contact with authors Dr. Thanh-Ha Do, and Dr. Duc-Tien Du. They sent an email link that was used to download a zip file which had the following pieces, a folder containing the following
This is a satellite dataset covering the period 2015–2023, using the IR1 spectral channel, resized to 625×500 pixels in PNG format:
0) The label information in the CSV format
1) for the geographical region: 
 biendong_startlat = -4.99
 biendong_startlon = 90.01
 biendong_endlat   = 34.99
 biendong_endlon   = 134.49
2) The dataset includes labels for several storm grades based on the [RSMC Tokyo best-track data](https://www.jma.go.jp/jma/jma-eng/jma-center/rsmc-hp-pub-eg/besttrack.html), which is also processed and attached in this mail. 
3) The original satellite data were obtained from [Chiba University, Japan](https://www.cr.chiba-u.jp/databases/GEO/H8_9/FD/index_en_V20190123.html). 

### Differences between the train and validation subsets, which are important from your project point of view
There are two strategies that I may use given the size and scope of this data set, the first is to simply use a random split of 60%-20%-20% for the data, which will hopefully ensure an ability for the model to categorize the location of the center of the TC at any point. Another would be to organize them by storm and use successive snapshots of the hurricane together. This relates to Adam's suggestion of adding in a temporal aspect to the computer vision algorithm.  With regards to time, the first will be tried to see a bulk estimation of the algorithm's effect, and then the use of individual tracks taken together will also be assessed.

### Number of distinct objects/subjects represented in the data, and number of samples per object/subject
This data set has 3,945 individual objects as images of a particular time from a satellite of a TC in the East Sea.  There are, as far as is evident, only one TC per image, with a corresponding best track location in latitude and longitude as well as a labeled bounding box of the center and a labeling of intensity. The first task of this project will be to recreate the storm center location, and then if given time, the intensity will also be predicted.

### Brief characterization of samples: resolution, sensors used, illumination wavelength, ambient conditions
As stated above, this data set uses the IR1 spectral channel, resized to 625×500 pixels in PNG format for the geographical region of the Biển Đông (East Sea, also known as the South China Sea). Since this is a static view of the sea that contains the TCs, it can be used easily for comparison across the different images.