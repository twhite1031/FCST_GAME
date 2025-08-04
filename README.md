# The Oswego Forecast Game
The python rendition of the SUNY Oswego forecast game played in Synoptic I and II
<br><br>
# Version
1.0
Refer to this repository for improvements and bug fixes
<br><br>
# Environment Installation
This project relies on several Python packages, so a Conda environment is provided for convenience.
<br><br>
Refer to [this](https://www.anaconda.com/docs/getting-started/miniconda/main) link if you are unfamiliar or do not have Miniconda/Anaconda
<br><br>
A conda environment (.yml) is provided [here](https://github.com/twhite1031/FCST_GAME/envs)
<br><br>
**Note**: Only a Windows environment has been currently been created, with linux planned in the near future. You could simply see the packages and make your own as well
<br><br>
After downloading the file, use this line within Anaconda prompt to create the environment in any directory:
```
conda env create -f FCST_GAME_win_1-0.yml
```
Allow around 10-15 minutes for the environment to create, conda can be painfully slow at times
Activate conda environment:
```
conda activate FCST_GAME
```
<br><br>
The environment can be activated anywhere within anaconda prompt and will remain activate until closing the software or using:
```
conda deactivate
```
# Cloning the Repository
To have this repository on your on system, simply use git within your environment:
```
git clone https://github.com/twhite1031/FCST_GAME/
```
# Preparing the Game
This game relies on two input files per day from Monday to Thursday. 
These are: m.24, m.48, t.24, t.48, w.24, w.48, r.24, r.48
You will also need a verification file for local station data. This can be substituted with your own data if you match the formatting.
This is: fcst.ver
**These files must be placed in input_data**

# Customizing the Game
Using your favorite text editor, open the synopticgame.py file found at the top level of the repository

In this file, you have will want to adjust the variables based on your input files or preference:
- Player names and identity (e.g. model or human)
- States that are being used in the flood game
- Precipitation threshold for verification

Then simply use python:
```
python synopticgame.py
```
# Output
There are two types of outputs regarding forecast numbers

## raw_data - data simply read from the files, no error calculations
- Consensus data for each file (csv)
## error_data - base error calculations, but no scaling or weights
- Consensus raw error for each file (csv)
- Forecaster raw error for each file (csv)
## final_output - full scaled error calculations
- Table of weighted error scores for all contestants (pdf)
- Consensus weighted error for entire week (csv)
- Forecaster weighted error for entire week (csv)

# Game rules and functionality
The forecast game rules for this game can be found [here](https://pi.cs.oswego.edu/~osscams/local_game_files/game_rules.jpg)

This script only uses the flood game portion, with storm and ice game planned for future release. <br>
ASOS data for verification are retrieved using the [Iowa State API](https://mesonet.agron.iastate.edu/api/)

# Change log
All notable changes to this project will be documented in this [file](https://github.com/twhite1031/FCST_GAME/CHANGELOG.md).
<br><br>
# Contact
For questions, bugs, or collaboration, feel free to reach out by opening an issue or contacting the maintainer.
<br><br>
Email: thomaswhite675@gmail.com or thomas.james.white@und.edu
