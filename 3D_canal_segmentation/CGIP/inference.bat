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