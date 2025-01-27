
import sys
import time
import math
import asyncio
import threading

sys.path.append('/home/seungwan/Desktop/wrapping/lib/python/x86_64/')
import robot_interface as sdk

def main():
    # if len(sys.argv) < 2:
    #     print(f"Usage: {sys.argv[0]} networkInterface")
    #     return -1
    
    # Assuming that the ChannelFactory and Init methods are properly bound in Python
    # sdk.ChannelFactory.Instance().Init(0, sys.argv[1])
    chan = sdk.ChannelFactory.Instance()
    chan.Init(0, "enx7cc2c64bce6b")

    # Initialize the sport client, assuming the translated classes have the same functionality
    sport_client = sdk.SportClient(False)
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    total_duration = 3  # Total duration is 3 seconds

    for i in range(1, total_duration + 1):
        if i <= 3:
            # Move forward for the first 3 seconds
            sport_client.Move(0.8, 0, 0)
        elif i > 3 and i <= 5:  # Corrected the logic to ensure this branch executes
            # Move left for the next 2 seconds
            sport_client.Move(0, 0.8, 0)
        else:
            # Move forward for the remaining time
            sport_client.Move(0.8, 0, 0)
        time.sleep(1)  # Wait for 1 second

    sport_client.StopMove()  # Stop moving

    return 0

if __name__ == "__main__":
    sys.exit(main())



'''
M_PI = 3.14159265358979323846

class Custom:
  c = 0
  dt = 0.002 ## 0.001~0.01

  def __init__(self):
    self.tc = sdk.SportClient(False)
    self.count = 0

  def control(self):
    self.c += 1
    
    vx = 0.3
    delta = 0.06
    self.count += self.dt

    path = []
    for i in range(30):
      p = sdk.PathPoint()
      var = (self.count + i * delta)
      p.timeFromStart = i * delta
      p.x = vx * var
      p.y = 0.6 * math.sin(M_PI * vx * var)
      p.yaw = 2*0.6 * vx * M_PI * math.cos(M_PI * vx * var)
      p.vx = vx
      p.vy = M_PI * vx * (0.6 * math.cos(M_PI * vx * var))
      p.vyaw = - M_PI * vx*2*0.6 * vx * M_PI * math.sin(M_PI * vx * var)
      path.append(p)


    ret = self.tc.TrajectoryFollow(path)
    print(f"{self.c}, {ret}") ## debugging
    if ret != 0:
      print(f"Call TrajectoryFollow: {ret}")

async def run_in_background(t ,func):
  while True:
    func()
    await asyncio.sleep(t)

def run_async_loop_in_thread(t, func):
    asyncio.run(run_in_background(t, func))

if __name__ == '__main__':

  chan = sdk.ChannelFactory.Instance()
  chan.Init(0, "")

  custom = Custom()
  custom.tc.SetTimeout(10.0)
  custom.tc.Init()
  t = threading.Thread(target=run_async_loop_in_thread, args=(1 * Custom.dt, custom.control))
  t.start()

  while True:
    time.sleep(10)
'''
