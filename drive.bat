@echo off
python %~dp0\is_model_old.py
IF %ERRORLEVEL%==1 (
    call activate udacity-self-driving-car
) ELSE (
    call activate saeed_local
)
::C:\ThirdEye\ase22\udacity-simulator\udacity-sim-win64\self_driving_car_nanodegree_program.exe
start /MAX C:\ThirdEye\ase22\udacity-simulator\self-compiled\self_driving_car_nanodegree_program.exe
python %~dp0\self_driving_car_drive.py
call conda deactivate