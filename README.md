datasciencecoursera
===================
This is public repository for Coursera Data Science Course

run_analysis.r file contains the reproducible code for generating tidy data set's 1 & 2 as per the course project details.

It assumes your data is under directory ".\UCI HAR Dataset"

ActivityLabels are loaded as Master Data set for associating activity labels with activities.

Train and Test data are loaded as separate data sets. These data sets are expanded with description of each activity from Activity Labels.

Variables of test and Train data sets are named as per feature list given. For sake of clarity and prevent syntax error max(), std(), etc are replaced 
with "_".

Finally a subset of columns is selected to create a tidy data set 1.