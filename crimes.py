import geopy
import time
import folium
import pandas as pd
import numpy as np
import statsmodels.api as sm
from folium import plugins
from haversine import haversine
from geopy.geocoders import Nominatim
from sklearn.metrics import hamming_loss
from sklearn.linear_model import LogisticRegression
'''
Read data sets (crime incidents, hospital locations, police station locaitons, liquor permits, public school locations in Boston) into data frames and pre-process (grab locations in latitudes and longitudes and other relavant information)
'''

model_start = time.time()
model_start2 = time.time()
start_time = time.time()

#Select which columns to read from csv file
columns = ['Location', 'WEAPONTYPE', 'Year']

#Parse crimesTrain.csv and pre-process
crimesTrainDF = pd.read_csv('crimesTrain.csv', usecols = columns)

#Split the 'Location' attribute into latitutde and longitude
crimesTrainDF['latitude'] = crimesTrainDF.Location.apply(lambda d: float(d.split(',')[0][1:]))
crimesTrainDF['longitude'] = crimesTrainDF.Location.apply(lambda d: float(d.split(',')[1][:-1]))

# Helper function to create DF column to represent armed or unarmed crime
def checkArmed(s):
	if s == 'Unarmed':
		return 0
	else:
		return 1

#Translate armed to 1 and unarmed to 0 and add to dataframe
crimesTrainDF['armed'] = crimesTrainDF.WEAPONTYPE.apply(lambda s: checkArmed(s))    

#Select which column to grab from csv
column = ['Location']

#Parse police station location data set and grab their locations
policeDF = pd.read_csv('stationTrain.csv', usecols = column)

#Split the 'Location' attribute into latitutde and longitude
policeDF['latitude'] = policeDF.Location.apply(lambda d: float(d.replace(' ', '').split('\n')[2].split(',')[0][1:]))
policeDF['longitude'] = policeDF.Location.apply(lambda d: float(d.replace(' ', '').split('\n')[2].split(',')[1][:-1]))

#Parse liquorTrain.csv and pre-process
liquorDF = pd.read_csv('liquorTrain.csv', usecols = column)

#Grab latitude and longitude from 'location' attribute
liquorDF['latitude'] = liquorDF.Location.apply(lambda d: float(d.split(',')[0][1:]))
liquorDF['longitude'] = liquorDF.Location.apply(lambda d: float(d.split(',')[1][:-1]))

#Parse hospital data set to find their location
hospitalDF = pd.read_csv('hospitalTrain.csv')
hospitalAD = hospitalDF['AD'].tolist()

#Parse public school locations data set and find latitude and longitudes
schoolDF = pd.read_csv('schoolTrain.csv', usecols = column)

#Extract longitude and latitude from address
schoolDF['latitude'] = schoolDF.Location.apply(lambda d: float(d.replace(' ', '').split('\n')[2].split(',')[0][1:]))
schoolDF['longitude'] = schoolDF.Location.apply(lambda d: float(d.replace(' ', '').split('\n')[2].split(',')[1][:-1]))

#Initialize lists of violent/non-violent crime, latitude and longitude
crimesArmed = crimesTrainDF['armed'].tolist()
crimesYear = crimesTrainDF['Year'].tolist()
crimesLat = crimesTrainDF['latitude'].tolist()
crimesLong = crimesTrainDF['longitude'].tolist()
cLoc = zip(crimesLat, crimesLong)

#Initialize list of tuples holding latitude and longitude of police station locations
policeLat = policeDF['latitude'].tolist()
policeLong = policeDF['longitude'].tolist()
pLoc = zip(policeLat, policeLong)

#Initialize list of tuples holding latitude and longitude of liquor permit locations
liquorLat = liquorDF['latitude'].tolist()
liquorLong = liquorDF['longitude'].tolist()
lLoc = zip(liquorLat, liquorLong)

#Initialize geolocator to find latitude and longitude hospitals using address
geolocator = Nominatim()

#iterate through addresses and grab latitude and longitude
hLoc = []
for i in range(len(hospitalAD)):    
    location = geolocator.geocode(hospitalAD[i])
    lat = location.latitude
    longs = location.longitude
    hLoc.append((lat, longs))

#Initialize list of tuples holding latitude and longitude of public schools
schoolLat = schoolDF['latitude'].tolist()
schoolLong = schoolDF['longitude'].tolist()
sLoc = zip(schoolLat, schoolLong)

print("Time to preprocess data %s seconds" % (time.time() - start_time))

'''
Pre-precessing complete! Now create some functions to process this data.
'''

#Takes in latitude and longitude of two locations and returns the distance between them in miles
def findProximity(lat1, long1, lat2, long2):
    location1 = (lat1, long1)
    location2 = (lat2, long2)
    return haversine(location1, location2, miles=True)

#Takes in location of a crime and then returns the proximity to the nearest police station, liquor permit, hospital and public school
def findClosest(latitude, longitude):
	#initialize values for later comparisons
    cPol = 100000000
    cLiq = 100000000
    cHos = 100000000
    cSch = 100000000
    #iterate through list of police station locations list
    for i in range(len(pLoc)):
        lats = pLoc[i][0]
        longs = pLoc[i][1]
        #find distance to closest police station locations
        temp = findProximity(latitude, longitude, lats, longs)
        if temp < cPol:
            cPol = temp
    #iterate through list of liquor permit locations
    for i in range(len(lLoc)):
        lats = lLoc[i][0]
        longs = lLoc[i][1]
        #find closest liquor licenses
        temp = findProximity(latitude, longitude, lats, longs)
        if temp < cLiq:
            cLiq = temp
    #iterate through list of hospital locations
    for i in range(len(hLoc)):
        lats = hLoc[i][0]
        longs = hLoc[i][1]
        #find closest hospital
        temp = findProximity(latitude, longitude, lats, longs)
        if temp < cHos:
            cHos = temp
    #iterate through list of public school locations
    for i in range(len(sLoc)):
        lats = sLoc[i][0]
        longs = sLoc[i][1]
        #find closest school
        temp = findProximity(latitude, longitude, lats, longs)
        if temp < cSch:
            cSch = temp
    return (cPol, cLiq, cHos, cSch)

'''
Great, now finally process the data: create a matrix X that holds proximity to the nearest police station, liquor permit, hospital and public school and a vector y to hold 1 or 0 depending on if the crime was violent or not
'''

start_time = time.time()

#Initialize lists to hold proximities to locations
pPol = []
pLiq = []
pHos = []
pSch = []

#Iterate through every crime and find proximity to police stations, liquor permit, hospital and public school locations
for i in range(len(cLoc)):
    lats = cLoc[i][0]
    longs = cLoc[i][1]
    temp = findClosest(lats, longs)
    pPol.append(temp[0])
    pLiq.append(temp[1])
    pHos.append(temp[2])
    pSch.append(temp[3])

print("Time to find proximities in training data: %s seconds " % (time.time() - start_time))

#Initialize lists to fit to a logistic regression
X = []
y = crimesArmed

#Create matrix X with each row of attributes
for i in range(len(pPol)):
    temp = []
    temp.append(pPol[i])
    temp.append(pLiq[i])
    temp.append(pHos[i])
    temp.append(pSch[i])
    temp.append(crimesYear[i])
    X.append(temp)

'''
Awesome, now fit the data into logistal regression model
'''

#Use the stats model api to fit the data to a logistic regression
model = sm.Logit(y, X)
results = model.fit()

#Print the summary of our logistial regression
print results.summary()

print "x1: proximity to police station"
print "x2: proximity to liquor permit"
print "x3: proximity to hospital"
print "x4: proximity to public school"
print "x5: year"
print 
print("Total time to create logistic regression results: %s seconds " % (time.time() - model_start))

'''
So now we have our model. Lets pre-process the data we want to predict
'''

start_time = time.time()

#Select which columns to read from csv file
columns = ['Location', 'WEAPONTYPE', 'Year']

#Parse crimesTrain.csv and pre-process
crimesPredictDF = pd.read_csv('crimesPredict.csv', usecols = columns)

#Split the 'Location' attribute into latitutde and longitude
crimesPredictDF['latitude'] = crimesPredictDF.Location.apply(lambda d: float(d.split(',')[0][1:]))
crimesPredictDF['longitude'] = crimesPredictDF.Location.apply(lambda d: float(d.split(',')[1][:-1]))

#Translate armed to 1 and unarmed to 0 and add to dataframe
crimesPredictDF['armed'] = crimesPredictDF.WEAPONTYPE.apply(lambda s: checkArmed(s))

#Initialize lists of violent/non-violent crime, latitude and longitude
crimesArmed2 = crimesPredictDF['armed'].tolist()
crimesYear2 = crimesPredictDF['Year'].tolist()
crimesLat2 = crimesPredictDF['latitude'].tolist()
crimesLong2 = crimesPredictDF['longitude'].tolist()
cLoc2 = zip(crimesLat2, crimesLong2)

# #Initialize lists to hold proximities to locations
pPol2 = []
pLiq2 = []
pHos2 = []
pSch2 = []

#Iterate through every crime and find proximity to police stations, liquor permit, hospital and public school locations
for i in range(len(cLoc2)):
    lats = cLoc2[i][0]
    longs = cLoc2[i][1]
    temp = findClosest(lats, longs)
    pPol2.append(temp[0])
    pLiq2.append(temp[1])
    pHos2.append(temp[2])
    pSch2.append(temp[3])

print("Time to find proximities in prediction data: %s seconds " % (time.time() - start_time))

#Initialize lists to fit to a logistic regression
X2 = []
y2 = crimesArmed2

#Create matrix X with each row of attributes
for i in range(len(pPol2)):
    temp = []
    temp.append(pPol2[i])
    temp.append(pLiq2[i])
    temp.append(pHos2[i])
    temp.append(pSch2[i])
    temp.append(crimesYear2[i])
    X2.append(temp)

print("Total time to create data for predictive model and pre-process predicton data: %s seconds " % (time.time() - model_start2))

'''
Yay! Now lets try to predict whether or not a crime will be violent or non-violent
'''

#Create logistic regression model and fit it to our training data X and y
pModel = LogisticRegression()
pModel.fit(X, y)

#Predict if a crime will be violent or non-violent with prediction data
yPredict = pModel.predict(X2)

#Print the predictions
print "Predictions:" + str(yPredict)

#Calculate the Hamming Loss to see how accurate our model is
print "Hamming loss: " + str(hamming_loss(y2, yPredict))

'''
Create some heat maps to visualize the data
'''

#Parse the violent crimes data set
violentCrimesDF = pd.read_csv('crimesViolent.csv')
violentCrimesDF = violentCrimesDF[['Location', 'Year', 'Month']]

#Create dates from July 2012 to April 2015 in 3 month intervals
yearmonth = [[2012, 7], [2012, 10], [2013, 1], [2013, 4], [2013, 7], [2013, 10], [2014, 1], [2014, 4], [2014, 7], [2014, 10], [2015, 1], [2015, 4]]

#Create lists of location, month and year
locations = violentCrimesDF['Location'].tolist()
month = violentCrimesDF['Month'].tolist()
year = violentCrimesDF['Year'].tolist()

#For every item in yearmonth, find all the locations of violent crimes occuring during that time frame
heatmaplocations = []
for item in yearmonth:
    heatmapinput = []
    for loc in range(len(locations)):
        if year[loc] == item[0] and month[loc] == item[1]:
            temp = locations[loc].strip('(').strip(')').replace(' ', '').split(',')
            heatmapinput.append([float(temp[0]), float(temp[1])])
    heatmaplocations.append(heatmapinput)

#Create heat maps using folium: save all 12 heatmaps as "heatap(i).html"
for i in range(len(heatmaplocations)):
    data = [[row[0], row[1]] for row in heatmaplocations[i]]
    hm = plugins.HeatMap(data)
    # Create a heatmap with the data centered in Boston, MA
    heatmap_map = folium.Map(location=[42.36455, -71.05796], zoom_start=14)
    heatmap_map.add_children(plugins.HeatMap(data))
    filename = "heatmap" + str(i) + ".html"
    heatmap_map.save(filename)