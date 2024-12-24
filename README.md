# COMP304_GroupProject
 
Centennial College <br/>
2024 Fall - COMP258 - Neural Network <br/>

## Table of Contents

+ [Facial Expression / Emotion Detection](#groupProject)
+ [Authors](#authors)

## Group Project <a name = "groupProject"></a>

Develop Facial Expression (Emotion detection) CNN MODEL <br/>
Includes <br/>
- Data Exploration
<img width="433" alt="image" src="https://github.com/user-attachments/assets/4e0dc76c-2040-4fa9-9316-38e886733215" />
<img width="433" alt="image" src="https://github.com/user-attachments/assets/b8926fdc-5e3d-445e-9197-eb130e052cdc" />
  
- Data Preprocessing <br/>
  • Verify data integrity <br/>
  • Ensure the correct format of images <br/>
  • Clean dataset without human faces <br/>
  • Data augmentation using ImageDataGenerator to balance the dataset and make them the same as 4000 examples for each class. <br/>
  • StratifiedShuffleSplit to shuffle the dataset and split the training and validation dataset. <br/>
  • Training data - (22400, 48, 48, 1) (64%) <br/>
  • Validation data - (5600, 48, 48, 1) (16%) <br/>
  • Testing data - (7176, 48,38,1) (20%) <br/>
  • Normalization via dividing by 255 <br/>
  
- CNN Model Architecture <br/>
  References: https://doi.org/10.54646/bijiam.2022.09
  
- Website Development 

## Authors <a name = "authors"></a>
- Patrick Tang - [@winghk00](https://github.com/winghk00)
- Yi-Chen Hsu - [@Janjanhsu](https://github.com/Janjanhsu)
