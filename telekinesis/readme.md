

```bash
Create 3 terminals

# First terminal: (Not use conda)
conda deactivate    && conda deactivate     
cd /media/yaxun/manipulation1/leaphandproject_ws
source devel/setup.zsh
roslaunch kinova_bringup kinova_robot.launch

# Second terminal: (Use conda)
cd /media/yaxun/manipulation1/leaphandproject_ws
conda activate bidex_manus_teleop
source devel/setup.zsh
python src/kinova_controller.py

# Third terminal: (Use conda)
ls /dev/ttyUSB*  && sudo chmod 777 /dev/ttyUSB*
adb devices

cd /media/yaxun/manipulation1/leaphandproject_ws
source devel/setup.zsh
conda activate bidex_manus_teleop
python src/telekinesis/leap_kinova_ik_real_recording.py     



# usb
sudo vim /etc/default/grub                                                                            
sudo update-grub   
# reboot
cat /sys/module/usbcore/parameters/usbfs_memory_mb


key:

r/p: recording /not
m/n: moving /not
s: succes
q: quit


workflow:
0. open
1. r: recording
2. m: moving arm
3. task
4. s: success (recording -> not)
5. reset pose (not recording)
6. goto 1
7. q: all tasks done

