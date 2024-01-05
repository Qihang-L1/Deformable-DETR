# **A compact version of Carla demonstrator**
---
> This is a general guide made by Tianming and Qihang which concludes their interpretation of the Capgemini 2d3dprojection and further code improvement regarding the intergration with 2d object detection algorithm. We also thank Stefan for his contribution to the docker establishment and program initialization part. 

---

## **Contents**  
*1. Environment configuration*  
*2. Program execution*  
*3. Issues about the I/O interface provided by Capgemini*    

---

## **environment configuration**
## 1) Build the CARLA docker image
```sh  
docker build -t carla_client
```

## 2) Create and run carla1 docker container in off-screen mode on the server
```sh
docker run -it --name carla1 --privileged --gpus all --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw carla_client bash
```

## 3) Prepare docker container for demonstrator 
- `ssh stu@10.200.4.19`
- `docker attach carla1`

### 3.a) Update AVL token in docker container
- Change the token in file: avl_hwp_ws/src/avl_hwp_common/scripts/avl_hwp_common_lib/interaction.py
- Token: "1dd3afa9-aa2b-433a-a384-e655d6e8d9fa"

### 3.b) Create a file "setup.bash" with the following content:
```sh
#!/bin/bash
export RENDER_MODE="-RenderOffScreen"
export delta_t="0.02"
# export mapname="/home/carla/xodr_files/Highway_OneWay_Curved_R450_LW2_8_Dashed.xodr"
# export lanesConfig="NOT_USED"
# export SCENARIO="/home/carla/scenarios/scenario_Highway_OneWay_Curved_R450_LW2_8_Dashed.xosc"
export mapname="/home/carla/xodr_files/Town04.xodr"
export lanesConfig="/home/carla/xodr_files/Town04_FreeRide_Highway_lanes.txt"
export SCENARIO="/home/carla/scenarios/Town04_FreeRide_Highway.xosc"
export module_converter="chmod +x /home/carla/avl_hwp_ws/src/avl_hwp_test_manually/scripts/main.py && source /opt/ros/melodic/setup.bash && source ./avl_hwp_ws/devel/setup.bash && rosrun avl_hwp_test_manually main.py"
export module_launchscript=""
export follow_ego="true"
export debug="false"
export use_splitted_stack="false"
export only_carla="false"
export visualize_paths="false"
export setup_hero_sensors="true"
export record_rosbag="false"
export exit_on_scenario_end="true"
```

 ---

## **Program execution**
## 1) Source all the setup file needed for ROS, Carla and the personal configuration.
***For every new terminal, you need to source it again!***
```sh
source /opt/ros/melodic/setup.bash && source /opt/carla-ros-bridge/melodic/setup.bash && source avl_hwp_ws/devel/setup.bash && source setup.bash
```
## 2) Run run.py
```sh
python3.8 run.py
```

- Clear all the processes after terminating this python script. Otherwise it will always occupies the GPU.
- We recommend restarting the container each time after demonstration. 

## 3) Demonstration on local computer
### 3.a) Create and run the carlaviz1 docker container on the server 
```sh
docker run -it --rm --name carlaviz1 --network="host" -e CARLAVIZ_BACKEND_HOST=localhost -e CARLA_SERVER_HOST=localhost -e CARLA_SERVER_PORT=2000 mjxu96/carlaviz:0.9.13 
```

### 3.b) Forward ports on your local machine (change stu to your user)
```sh
ssh -L 8080:10.200.4.19:8080 -L 8081:10.200.4.19:8081 -L 8089:10.200.4.19:8089 -f -N stu@10.200.4.19
```

### 3.c) Open browser at
- `localhost:8080`

### 3.d)Reset port when you are done with the entire process
```sh
lsof -ti:8080 | xargs kill -9
```

## 4) Run 2d/3d object converter
### 4.a) Create new entry point to container
```sh
docker exec -it carla1 bash
```

### 4.b) Download capgemini_3d_objects_tracking
```sh
git clone https://gitlab.com/ki-wissen/tp4/ap4.2/capgemini_3d_objects_tracking.git
```
### 4.c) Update PYTHONPATH
```sh
export PYTHONPATH="/home/carla/capgemini_3d_objects_tracking":"/home/carla/capgemini_3d_objects_tracking/multipleobjecttracking":${PYTHONPATH}
```

### 4.d) Source everything
```sh
source /opt/ros/melodic/setup.bash && source /opt/carla-ros-bridge/melodic/setup.bash && source avl_hwp_ws/devel/setup.bash && source setup.bash
```
### 4.e) Run the converter
```sh
python3.8 capgemini_3d_objects_tracking/projection2d3d.py
```

- You should now see the following two topics with "rostopic list":
    ```
    /CAM_0/prior/image_raw
    /inference/hero/objects
    ```

---

## **Issues about the I/O interface provided by Capgemini**
## 1) ROS node structure
- In the initial file provided by Capgemini, there is only one python script and several other files defining functions used by the main program. Now we create an individual workspace for the 2d3d branch, along with all the basic ROS structure where you can add dependency between packages, edit CMakelist and write roslauch file. 

## 2) Inputs for the object detection 
- In projection2d3d.py, they defined a function called "AI_2D_ObjectDetection" where we suppose to integrate our own codes. The primitive data we receive is the rgb message and tf message subscribed by the rosnode. This cannot be fed directly to the following object detection algorithm. 

- Hence, we add some codes to convert the ROS image message into the 4 dimension numpy arrays that can be processed by Pytorch:
```sh
data = rgb_message
cv_image = np.frombuffer(data.data, np.uint8).reshape((data.height, data.width, 4))
```

- The cv_bridge library is not needed when running the codes above.

## 3) Outputs required by the 2d3dprojection
- Apart from the 2d booundingbox array and class_id array, the object detection needs to provide mask for the convertion, which is a precise specification of the object inside each boundingbox. However, this is not a necessity.

- In projection2d3d.py, they defined two publishers to publish the 3d boundingbox message. Publisher2 is for the following AD stack control part and publisher3 for Carla simulator to receive the image information so that it can demonstrate the real-time boundingboxes in camera view.

- A new subscriber in setup_hero_sensors.py can be created to subcribe to publisher3.

## 4) Thread problem
- In projetion2d3d.py, the main function "start" acts as a subscriber node to receive all the information needed from the callback functions, while the function "inference" deals with the processing and publishing of the information. 

- Because of the frame by frame processing, the publisher thread and the subscriber thread must be split and coordinate with each other. However, the initial code from Capgemini cannot be run properly because the code rospy.spin() will terminate the process without waiting for the inference thread to join. 

- We add a new thread to run this code seperately and solve this problem:
```sh
t1 = threading.Thread(target = inference)
t2 = threading.Thread(target = thread_spin)
t1.start()
t2.start()
t1.join()
t2.join()
```

```sh
def thread_spin()
    rospy.spin()
```