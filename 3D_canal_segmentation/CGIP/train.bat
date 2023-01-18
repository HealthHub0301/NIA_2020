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