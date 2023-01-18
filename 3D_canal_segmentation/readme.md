# Mandibular Canal segmentation:
This section explains the implementation of mandicular canal segmentation by deep learning model. There are two files one for training and other of taking inference.

# Data structuring.
Data have to structured in following way There have to be two directories one for training and other for testing.
Training directories have sub directories which contain training samples.
Each subdirectory should have dicom slices and .json annotation for that samples.
Similar structure is addopted for test dataset.

```
----NIA 2020:
	│
	├── Training
	│   └── 0001
	│       └── 0001.json
	│	└── 001457510126.dcm
	│	└── 001457510127.dcm
	│	└── 001457510128.dcm
	│	└── ....
	│   └── 0002
	│   └── 0003
	│   └── 0004
	│   └── .......
	│
	├── Testing
	│   └── 0901
	│       └── 0901.json
	│	└── 001457510126.dcm
	│	└── 001457510127.dcm
	│	└── 001457510128.dcm
	│	└── ....
	│   └── 0902
	│   └── 0903
	│   └── 0904
	│   └── .......
```

# Running training file.
Training file will train model on training samples and save left or right weights for model depending on parameter given at input.
   run:
   ```shell
   python training_model.py --data_path='NIA 2020/Training' --side='left'
   ```

# Running inference file.
Inference file will save a json file name 0901 with annotations for mask and will make a results.csv and append quantitative scores at the end of file.
   run:
   ```shell
   python Testing_file.py --scan_path='NIA 2020/Testing/0901' --mask_path='NIA 2020/Testing/0901'
   ```

# CGIP:
3D segmentation from CGIP

Inference(inference.bat) : 
@ECHO off
SET prj_name=%1

SET model=e:\Data\Teeth-Pano\models\%prj_name%\best.pth
SET data_path=e:\Data\Teeth-Pano\test
SET dst=e:\Data\Teeth-Pano\results\%prj_name%

MKDIR %dst%

SET python_venv=D:\Anaconda3\envs\Teeth-Pano\python.exe

%python_venv% inference.py 	--model %model%^
							--data_path %data_path%^
							--dst %dst%^
							--save_gt
							
						
:: Evaluate
%python_venv% eval.py		--data_dir %data_path%^
							--result_dir %dst%
							
Train(train.bat):
@ECHO off
SET prj_name=%1

SET data_path=e:\Data\Teeth-Pano
SET pretrained_model_path=e:\Data\Teeth-Pano\models\

:: Save Model
SET model_save_path=e:\Data\Teeth-Pano\models\%prj_name%
SET log_path=%model_save_path%\logs

MKDIR %model_save_path%
MKDIR %log_path%


:: Run a model
SET python_venv=D:\Anaconda3\envs\Teeth-Pano\python.exe

%python_venv% train.py 	--prj_name %prj_name%^
                        --data_path %data_path%^
						--log_path %log_path%^
						--model_save_path %model_save_path%^
						--pretrained_model_path %pretrained_model_path%^
						--batch_size 5^
						--backbone dla^
						--use_center
						
						
