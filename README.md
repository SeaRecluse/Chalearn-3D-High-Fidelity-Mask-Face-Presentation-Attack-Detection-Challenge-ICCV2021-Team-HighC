# Chalearn 3D High-Fidelity Mask Face Presentation Attack Detection Challenge@ICCV2021 —— Team HighC


## Infos
The backbone network comes from https://github.com/Randl/ShuffleNetV2-pytorch



## Step
Install dependencies:
```bash
pip install -r requirements.txt
```

Data preprocessing:
If you have full content data data, it should look like this in the folder
--orig_data
    -train *
    -test *
    -val *
    -train_label.txt *
    -val_label.txt *
    -test.txt *
    -val.txt *
    -getFacesList.py
    -sortDataWithLabel.py
    -dataUpper.py
    -cut.py
    -detect_face_wIth_align

```
// Build a folder to save face alignment results
// For pictures with no faces detected, manual calibration is recommended
use getFacesList.py

// we use retinaface .25 to detect face and do face align
use main.py in /orig_data/detect_face_wIth_align

// Integrate your face alignment data
use sortDataWithLabel.py

// Data enhancement
use dataUpper.py

// Divide your data set(Just use it in Phase I)
use cut.py
```

Put the enhanced data into the folder "./data" for training
```
python main.py
```

Test your model and draw ROC performance graph
```
python test.py
cd ./draw
./show.bat

// if you want to test all test data
// change ONLY_VAL = 0 in test.py
```

## others
We found that the final score is closely related to the accuracy of face alignment

