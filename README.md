# Neuroimgaing processing library
## Requirements
* Downloaded [SPM](http://www.fil.ion.ucl.ac.uk/spm/ext/) toolbox

##Processing data  
The analysis pipeline can be described as following:
**Rearranging folders** -> **Processing images** -> **Extracting features** -> Apply machine learning algorithms for **classification**
See more details of implementation in the wiki.
In order to start analysis `Main_script` both for Matlab and Python has to be used.


## Results for current repopsitory state  
**(1) - Male**  
**(2) - Female**  
**AV45 images Processing results**  

|         |  Total | mean_Age(1) | Std_Age(1) | MMSE(1) | mean_Age(2) |Std_Age(2) | MMSE(2)|
| :-----: | :-----:|:---: |:---:| :---:| :---:| :---:| :---:|
| Normal  | 408    | 74.13|6.08| 28.67|72.26 | 5.68 | 28.96|
| MCI      | 87    | 75.2|7.16 | 27.59 |75.32 |5.2 | 27.48|
| AD      | 237    | 75.46|7.45 | 21.23 |72.75 |7.65  | 21.09|  


**Test dataset size**: 74  
Confusion matrix
|        | Normal| MCI   | AD   |
| :----: | ----: | :---: |:---: |
| Normal | 35 |  0  | 4     |
| MCI    | 5  |  3  | 1     |
| AD     | 2  |  0  | 24    |

Best **mean** accuracy for 10 folds: **0.862**

--------

**FDG images Processing results**  

|         |  Total | mean_Age(1) | Std_Age(1) | MMSE(1) | mean_Age(2) |Std_Age(2) | MMSE(2)|
| :-----: | :-----:|:---: |:---:| :---:| :---:| :---:| :---:|
| Normal  | 283    | 74.11|5.9|  28.7|72.04 | 6.05 | 29.02|
| MCI      | 57    | 76.23|6.9 | 27.5 |75.24 |5.54  | 27.46|
| AD      | 202    | 75.50|7.63| 21.04   |72.59 |7.66  |  21.06|  


**Test dataset size**: 55
![Confusion matrix](/Results/fdg_confusion_matrix.png)

Best **mean** accuracy for 10 folds: **0.815**
