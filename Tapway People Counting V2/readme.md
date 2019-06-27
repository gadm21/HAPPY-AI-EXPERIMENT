# Tapway Table Occupancy Monitoring
``` Status: In Development ```

## Prerequisites:
1) Install Anaconda

a. Windows
- Anaconda 2019.03 (https://repo.anaconda.com/archive/Anaconda3-2019.03-Windows-x86_64.exe)
- Full installation guide: https://docs.anaconda.com/anaconda/install/

b. Linux
- Anaconda 2019.03 (https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh)
- Full installation guide: https://docs.anaconda.com/anaconda/install/

2) Setting up Anaconda
- Create an environment with the following commands:
  ``` conda install -c anaconda tensorflow-gpu ```
  ``` refer to document written in client server ```
  ``` x ```
  ``` x ```
  ``` x ```
  ``` x ```
  
3) MySQL Database
- id (primary key) int
- table_id int
- dwell_time int

### How to run people_counting.py
- "py people_counting.py" on Windows, "python3 people_counting.py" on Linux

### How to get_roi_points.py
- "py get_roi_points.py" on Windows, "python3 get_roi_points.py" on Linux
- enter the name of the selected camera
- press on the pop up windows, then press "r"
- draw a box to locate the table (draw again if not satisfied)
- press "enter"
- press "q" to confirm
- the ROI points are displayed on the command prompt
- you may now insert the table_id and ROI points into configuration JSON file

### Notes for JSON Configuration
- Remember to change "num_of_tables" whenever you add a table
