# Neuroimgaing processing library
## Requirements
* Downloaded [SPM](http://www.fil.ion.ucl.ac.uk/spm/ext/) toolbox

## Processing data  
The analysis pipeline can be described as following:
**Rearranging folders** -> **Processing images** -> **Extracting features** -> Apply machine learning algorithms for **classification**
See more details of implementation in the wiki.
In order to start analysis `Main_script` both for Matlab and Python has to be used.


## Results for current repopsitory state  
**(1) - Male**  
**(2) - Female**  
**AV45 images Processing results**  

<table><tr><th rowspan="2"><br></th><th rowspan="2">Total</th><th colspan="2">Mean Age</th><th colspan="2">Std Age</th><th colspan="2">MMSE</th></tr><tr><td>Male</td><td>Female</td><td>Male</td><td>Female</td><td>Male</td><td>Female</td></tr><tr><td>Normal</td><td>408</td><td>74.13</td><td>72.26</td><td>6.08</td><td>5.68</td><td>28.67</td><td>28.96</td></tr><tr><td>MCI</td><td>87</td><td>75.2</td><td>75.32</td><td>7.16</td><td>5.2</td><td>27.59</td><td>27.48</td></tr><tr><td>AD</td><td>237</td><td>75.46</td><td>72.75</td><td>7.45</td><td>7.65</td><td>21.23</td><td>21.09</td></tr></table>

**Test dataset size**: 74, where: Normal: 39; MCI: 9; AD: 26    
    
|Confusion Matrix|Normilized CM|
|:----:|:----:|
|<table><tr><th><br></th><th>Normal</th><th>MCI</th><th>AD</th></tr><tr><td>Normal</td><td>35</td><td>0</td><td>4</td></tr><tr><td>MCI</td><td>5</td><td>3</td><td>1</td></tr><tr><td>AD</td><td>2</td><td>0</td><td>24</td></tr></table>|<table><tr><th><br></th><th>Normal</th><th>MCI</th><th>AD</th></tr><tr><td>Normal</td><td>0.9</td><td>0</td><td>0.1</td></tr><tr><td>MCI</td><td>0.5</td><td>0.33</td><td>0.11</td></tr><tr><td>AD</td><td>0.08</td><td>0</td><td>0.92</td></tr></table>|

Best **mean** accuracy for 10 folds: **0.862**

--------

**FDG images Processing results**  

<table><tr><th rowspan="2"><br></th><th rowspan="2">Total</th><th colspan="2">Mean Age</th><th colspan="2">Std Age</th><th colspan="2">MMSE</th></tr><tr><td>Male</td><td>Female</td><td>Male</td><td>Female</td><td>Male</td><td>Female</td></tr><tr><td>Normal</td><td>238</td><td>74.11</td><td>72.04</td><td>5.9</td><td>6.05</td><td>28.7</td><td>29.02</td></tr><tr><td>MCI</td><td>57</td><td>76.23</td><td>75.24</td><td>6.9</td><td>5.54</td><td>27.5</td><td>27.46</td></tr><tr><td>AD</td><td>202</td><td>75.5</td><td>72.6</td><td>7.63</td><td>7.66</td><td>21.04</td><td>21.06</td></tr></table> 


**Test dataset size**: 55, where: Normal: 34; MCI: 7; AD: 14  
    
|Confusion Matrix|Normilized CM|
|:----:|:----:|
|<table><tr><th><br></th><th>Normal</th><th>MCI</th><th>AD</th></tr><tr><td>Normal</td><td>29</td><td>1</td><td>4</td></tr><tr><td>MCI</td><td>3</td><td>3</td><td>1</td></tr><tr><td>AD</td><td>4</td><td>0</td><td>10</td></tr></table>|<table><tr><th><br></th><th>Normal</th><th>MCI</th><th>AD</th></tr><tr><td>Normal</td><td>0.85</td><td>0.03</td><td>0.12</td></tr><tr><td>MCI</td><td>0.43</td><td>0.43</td><td>0.14</td></tr><tr><td>AD</td><td>0.28</td><td>0</td><td>0.71</td></tr></table>|

Best **mean** accuracy for 10 folds: **0.815**
