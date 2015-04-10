from ggplot import *
import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from PIL import Image
import numpy as np


filepath = '../turnstile_weather_v2.csv'
turnstile_weather = pandas.read_csv(filepath)


def entries_histogram(turnstile_weather):
    
    plt.figure()

    plt.xlabel('hourly entries')
    plt.ylabel('frequency')
    #plot a historgram for hourly entries when it is not raining
    turnstile_weather[turnstile_weather['rain']==False]["ENTRIESn_hourly"].hist(bins=80, label = "no rain")

    #plot a historgram for hourly entries when it is raining
    turnstile_weather[turnstile_weather['rain']==True]["ENTRIESn_hourly"].hist(bins=80, label = "rain")

    #put in a legend...
    plt.legend()

    plt.show()
    return


## generate a bitmap image, where the horizontal axis represents the 4 hour periods
## over the month of may, and the vertical axis reprsents each individual turnstile collection unit.
def rain_by_time_and_station(turnstile_weather):

	##Obtain a list of each of the turnstile units
	units = list(set(turnstile_weather['UNIT'].values.tolist()))

	## create a dictionary of dates to numerical values,
	## to keep track of order, and fill in blanks

	dates = list(set(turnstile_weather['DATEn'].values.tolist()))
	#print dates
	dates_dict = {}
	i=0 
	for date in sorted(dates):
		dates_dict[date]=i
		i+=1
	#print dates_dict

	time_dict = {
	"00:00:00":0,
	"04:00:00":1,
	"08:00:00":2,
	"12:00:00":3,
	"16:00:00":4,
	"20:00:00":5	
	}

	print 'step 1 complete'
	

	## iterate through data for each unit,
	## and assemble 
	rain_by_unit = []
	for unit in units:
		data = turnstile_weather[turnstile_weather['UNIT']==unit][["DATEn","TIMEn","rain"]]
		rain_by_date = [2]*(6*len(dates_dict))
		#print len(rain_by_date)
		for entry in data.iterrows():
			pos = dates_dict[entry[1]["DATEn"]]*6+time_dict[entry[1]["TIMEn"]]
			#print entry[1]["DATEn"]
			#print pos
			if entry[1]["rain"]:
				rain_by_date[pos] = 1
			else:
				rain_by_date[pos] = 0
			
		#print len(rain_by_date)
		rain_by_unit.append(rain_by_date)


	
	
	
	print 'step 2 complete'

	## create an bitmap image to represent the data 
	## and write it to 'rain_by_unit_and_date.bmp'
	height = len(rain_by_unit)
	width = len(rain_by_unit[0])
	img = Image.new('RGB', (width,height), "black") # create a new blank image
	pixels = img.load()
	#print height
	#print width
	#print img.size

	## here i is the index for date/time and j is the index for turnstile unit
	for i in range(img.size[0]):
		for j in range(img.size[1]):
			#print i
			#print j
			if rain_by_unit[j][i] == 1:
				pixels[i,j] = (255,255,255)
			elif rain_by_unit[j][i]==0:
				pixels[i,j] = (0,0,0)
			else:
				pixels[i,j] = (255,0,0)
	print 'step 3 complete'	
	
	img.save("../rain_by_unit_and_date","bmp")


def ridership_by_time(turnstile_weather):

	## create a dictionary of dates to numerical values,
	## to keep track of order, and fill in blanks

	dates = list(set(turnstile_weather['DATEn'].values.tolist()))
	#print dates
	dates_dict = {}
	i=0 
	for date in sorted(dates):
		dates_dict[date]=i
		i+=1
	#print dates_dict

	time_dict = {
	"00:00:00":0,
	"04:00:00":1,
	"08:00:00":2,
	"12:00:00":3,
	"16:00:00":4,
	"20:00:00":5	
	}

	print 'step 1 complete'
	

	## iterate through data for each unit,
	## and enter
	riders_by_date = [0]*(6*len(dates_dict))
	#print len(rain_by_date)
	for entry in turnstile_weather.iterrows():
		pos = dates_dict[entry[1]["DATEn"]]*6+time_dict[entry[1]["TIMEn"]]

		riders_by_date[pos] += entry[1]["ENTRIESn_hourly"]
			
	#print len(rain_by_date)
	
	print 'step 2 complete'

	plt.plot(riders_by_date)
	plt.show()
	
	




#entries_histogram(turnstile_weather)

#rain_by_time_and_station(turnstile_weather)

ridership_by_time(turnstile_weather)
