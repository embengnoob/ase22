@echo off
set arg1=%1
call activate saeed_local
python %~dp0\self_driving_car_train.py -es %arg1%
