@echo off
call activate udacity-self-driving-car
start C:\ThirdEye\ase22\udacity-simulator\udacity-sim-win64\self_driving_car_nanodegree_program.exe
python %~dp0\self_driving_car_drive.py
