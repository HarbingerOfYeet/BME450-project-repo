# BME450 Machine Learning Project
	Age Prediction from Vocal Input
## Team Members
	Winston Ngo (HarbingerOfYeet), Andrew Wacker (ahwacker), Christopher Chang (ChrisC26)
## Project Description
	The goal of this project is to predict a user's age based on their vocal patterns. The dataset we plan to use is Mozilla's
	Common Voice dataset. Common voice is an online database of thousands of voices that is commonly used for speech recognition
	purposes. Two stretch goals are to be able to conduct real-time age detection and predict the user's sex in addition to age. 
	We plan to use an MLP to train on the dataset. 

	Process:
		1. Download Common Voice Delta Segment 12.0 and select audio files with a non-empty age field
		2. Determine MFCC for each audio file
		3. Run 100 audio clips through MLP model to train on a small dataset
		4. Test model 

	Common Voice: https://commonvoice.mozilla.org/en
