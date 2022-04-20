#### Header ####
## Big Data 
## Harry Buckley - 17239400
## last edited: 20/04/2022


# QUICK SCRIPT FOR MAKING ANIMATED MAP

#data loading
aus <- readr::read_csv("https://raw.githubusercontent.com/BUCKERS99/C7084-Big-data--Will-it-rain-tomorrow-/main/weather_data.csv")

## EDA
# As the dependent variable is going to be the binomial answer to rain tomorrow? We need to remove and missing values.
sum(is.na(aus$RainTomorrow))
# there are 3267 NA values so these rows need to be removed. use the na.omit function to create a new df
tidy_aus <- aus[!is.na(aus$RainTomorrow),]

# check that the number of rows removed are equal to 3267
print(nrow(aus) - nrow(tidy_aus))

# Variables need to be correctly classed
tidy_aus$Date <- as.Date(tidy_aus$Date, tryFormats = "%d/%m/%y")
tidy_aus$Location <- as.factor(tidy_aus$Location)
tidy_aus$RainToday <- as.factor(tidy_aus$RainToday)
tidy_aus$RainTomorrow <- as.factor(tidy_aus$RainTomorrow)
# R automaticall changes between number and integer during operations so this is not a problem. 
tidy_aus$Location
levels(tidy_aus$Location)


# making an interactive map plot of Australia

# if require in the set up not always working....
library(readxl)
library(tidyverse)
library(dplyr)

#read in the location lat long file from github
locs <- read_csv("https://raw.githubusercontent.com/BUCKERS99/C7084-Big-data--Will-it-rain-tomorrow-/main/Location_lats.csv")
locs$Location <- as.factor(locs$Location)

# join the two data sets
tidy_aus <- tidy_aus %>% right_join(locs, by = "Location")

# select the variable we want to use
map_aus <- tidy_aus %>%
  select(Date, Location, RainTomorrow, Lat, Long)

# set them as factors incase this hasnt worked
map_aus$Location <- as.factor(map_aus$Location)
map_aus$RainTomorrow <- as.factor(map_aus$RainTomorrow)




# using ggplot
library(ggplot2)
library(gganimate)
library(gifski)
library(ozmaps)

#getting the outline of australia
oz_states <- ozmap_states
# this allows the map to plot points on
sf::sf_use_s2(FALSE)

# make a base map
base_map <- ggplot(oz_states) +
  geom_sf() +
  geom_point(map_aus, mapping = aes(Long, Lat, color = RainTomorrow)) +
  coord_sf()
base_map

# animate the base map
map_animate <- base_map +
  transition_time(Date) +
  ggtitle('Date: {frame_time}',
          subtitle = 'Frame {frame} of {nframes}')
num_date <- max(map_aus$Date) - min(map_aus$Date) +1
animate(map_animate, nframes = num_date, fps = 2)

# anim_save("Rain_Tomorrow.gif")
