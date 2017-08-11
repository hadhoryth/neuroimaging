# Neuroimgaing processing library
## Requirements
* Downloaded [SPM](http://www.fil.ion.ucl.ac.uk/spm/ext/) toolbox

--------
The analysis pipeline can be described as following:  
**Rearranging folders** -> **Processing images** -> **Extracting features** -> Apply machine learning algorithms for **classification**.  
See more details of implementation in the wiki.  
In order to start analysis `Main_script` both for Matlab and Python has to be used.
--------

## Results for current repopsitory state
**(1) - Male**
**(2) - Female**
**AV45 images Processing results**

|         |  Total | mean_Age(1) | Std_Age(1) | MMSE(1) | mean_Age(2) |Std_Age(2) | MMSE(2)|
| :-----: | :-----:|:---: |:---:| :---:| :---:| :---:| :---:|
| Normal  | 374    | 74.39|5.89| 28.73|72.26 | 5.68 | 28.96|
| AD      | 224    | 76.50|7.38 | 23.22 |74.23 |6.9  | 22.3|

Best **mean** accuracy for 10 folds: **0.892**

--------

**FDG images Processing results**

|         |  Total | mean_Age(1) | Std_Age(1) | MMSE(1) | mean_Age(2) |Std_Age(2) | MMSE(2)|
| :-----: | :-----:|:---: |:---:| :---:| :---:| :---:| :---:|
| Normal  | 268    | 74.26|5.82|  28.71|72.39 | 5.71 | 28.98|
| AD      | 194    | 76.45|7.51| 22.6   |74.039 |7.234  |  21.89|

Best **mean** accuracy for 10 folds: **0.84**
