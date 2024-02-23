@echo off
python %~dp0\is_model_old.py
IF %ERRORLEVEL%==1 (
    call activate udacity-self-driving-car
) ELSE (
    call activate saeed_local
)
::C:\ThirdEye\ase22\udacity-simulator\udacity-sim-win64\self_driving_car_nanodegree_program.exe
start /MAX %~dp0\udacity-simulator\self-compiled\maxspeed30mph_save_position_in_training_night_only\self_driving_car_nanodegree_program.exe
python %~dp0\self_driving_car_drive.py
call conda deactivate