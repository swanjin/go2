#include <unitree/robot/go2/sport/sport_client.hpp>
#include <unitree/idl/go2/SportModeCmd_.hpp>
#include <unitree/common/thread/recurrent_thread.hpp>
#include <unitree/robot/go2/obstacles_avoid/obstacles_avoid_client.hpp>
#include <math.h>
#include <unistd.h>
#include <iostream>
#include <unitree/idl/ros2/PointCloud2_.hpp>

class Custom
{
public:
  Custom() {}
  void control();

  unitree::robot::go2::SportClient tc;

  int c = 0;
  float dt = 0.002; // 0.001~0.01
};

void printPosition(const std::array<float, 2>& position) {
    std::cout << "Position - X: " << position[0] << ", Y: " << position[1] << std::endl;
}

void Custom::control()
{
    c++;
    
    int32_t ret;

    float vx = 0.3;
    float delta = 0.06;
    static float count = 0;
    count += dt;
    std::vector<unitree::robot::go2::PathPoint> path;
    for (int i=0; i<30; i++) {
      unitree::robot::go2::PathPoint p;
      float var = (count + i * delta);
      p.timeFromStart = i * delta;
      p.x = vx * var;
      p.y = 0.6 * sin(M_PI * vx * var);
      p.yaw = 2*0.6 * vx * M_PI * cos(M_PI * vx * var);
      p.vx = vx;
      p.vy = M_PI * vx * (0.6 * cos(M_PI * vx * var));
      p.vyaw = - M_PI * vx*2*0.6 * vx * M_PI * sin(M_PI * vx * var);
      path.push_back(p);
    }
    std::cout << "Sending trajectory follow command, iteration: " << c << std::endl;

    ret = tc.TrajectoryFollow(path);
    if(ret == 0){
        std::cout << "TrajectoryFollow command was successful." << std::endl;
    } else {
        std::cout << "Call to TrajectoryFollow failed with error: " << ret << std::endl;
    }
}

int main(int argc, char** argv)
{
  unitree::robot::ChannelFactory::Instance()->Init(0);

  Custom custom;
  custom.tc.SetTimeout(10.0f);
  custom.tc.Init();

  unitree::common::ThreadPtr threadPtr = unitree::common::CreateRecurrentThread(custom.dt * 1000000, std::bind(&Custom::control, &custom));

  while (1)
  {
    sleep(10);
  }

  return 0;
}

// Simple Command

/*int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " networkInterface" << std::endl;
        return -1;
    }
    
    unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);

    // Initialize SportClient
    unitree::robot::go2::SportClient sport_client;
    sport_client.SetTimeout(10.0f);
    sport_client.Init();

    const int totalDuration = 7; // Duration = 7

    for(int i = 1; i <= totalDuration; ++i){
        if(i <= 3) {
            // Move forward for 3 secs
            sport_client.Move(0.8, 0, 0);
        } else if(i <= 2) {
            // Move forward for 2 secs
            sport_client.Move(0, 0.8, 0);
        } else {
            // Move forward
            sport_client.Move(0.8, 0, 0);
        }
        sleep(1);
    }

    sport_client.StopMove();

    return 0;
}*/

// Obstacle Avoidance Checking

/*
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " networkInterface" << std::endl;
        return -1;
    }
    
    unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);

    // Obstacles Avoid Client initialization and activation
    unitree::robot::go2::ObstaclesAvoidClient obstacles_avoid_client;
    obstacles_avoid_client.Init();
    obstacles_avoid_client.SwitchSet(true);

    // SportClient initialization
    unitree::robot::go2::SportClient sport_client;
    sport_client.SetTimeout(10.0f);
    sport_client.Init();

    bool obstacleDetected = false;
    int duration = 15; // total duration time

    for(int i = 0; i < duration; ++i){
        obstacles_avoid_client.SwitchGet(obstacleDetected);
        if(obstacleDetected){
            // if obstacle is detected, move to left
            sport_client.Move(0, 0.8, 0);
        }else{
            // else, move to forward
            sport_client.Move(0.8, 0, 0);
        }
        sleep(1);
    }

    sport_client.StopMove();

    return 0;
}*/
