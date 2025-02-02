#include <unitree/robot/go2/sport/sport_client.hpp>
#include <unitree/robot/client/client_base.hpp>
#include <unitree/robot/client/client.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <iostream>
#include <unistd.h>
#include <string.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace unitree;
using namespace unitree::robot;
using namespace unitree::robot::go2;

namespace py = pybind11;

PYBIND11_MODULE(robot_interface, m) {
    // sport_client.hpp
  py::class_<PathPoint>(m, "PathPoint")
      .def(py::init<>())
      .def_readwrite("timeFromStart", &PathPoint::timeFromStart)
      .def_readwrite("x", &PathPoint::x)
      .def_readwrite("y", &PathPoint::y)
      .def_readwrite("yaw", &PathPoint::yaw)
      .def_readwrite("vx", &PathPoint::vx)
      .def_readwrite("vy", &PathPoint::vy)
      .def_readwrite("vyaw", &PathPoint::vyaw);

  py::class_<ClientBase> clientBase(m, "ClientBase");
    clientBase
      .def("Init", &ClientBase::Init)
      .def("SetTimeout", py::overload_cast<int64_t>(&ClientBase::SetTimeout))
      .def("SetTimeout", py::overload_cast<float>(&ClientBase::SetTimeout));
      
  py::class_<Client, ClientBase> client(m, "Client");
    client
      .def("WaitLeaseApplied", &Client::WaitLeaseApplied)
    //   .def("GetLeaseId", &Client::GetLeaseId)
      .def("GetApiVersion", &Client::GetApiVersion)
      .def("GetServerApiVersion", &Client::GetServerApiVersion);

  py::class_<SportClient, Client>(m, "SportClient")
      .def(py::init<bool>())
      .def("Init", &SportClient::Init)
      .def("Damp", &SportClient::Damp)
      .def("BalanceStand", &SportClient::BalanceStand)
      .def("StopMove", &SportClient::StopMove)
      .def("StandUp", &SportClient::StandUp)
      .def("StandDown", &SportClient::StandDown)
      .def("RecoveryStand", &SportClient::RecoveryStand)
      .def("Euler", &SportClient::Euler)
      .def("Move", &SportClient::Move)
      .def("Sit", &SportClient::Sit)
      .def("RiseSit", &SportClient::RiseSit)
      .def("SwitchGait", &SportClient::SwitchGait)
      .def("Trigger", &SportClient::Trigger)
      .def("BodyHeight", &SportClient::BodyHeight)
      .def("FootRaiseHeight", &SportClient::FootRaiseHeight)
      .def("SpeedLevel", &SportClient::SpeedLevel)
      .def("Hello", &SportClient::Hello)
      .def("Stretch", &SportClient::Stretch)
      .def("TrajectoryFollow", &SportClient::TrajectoryFollow)
      .def("SwitchJoystick", &SportClient::SwitchJoystick)
      .def("ContinuousGait", &SportClient::ContinuousGait)
      .def("Wallow", &SportClient::Wallow)
      .def("Heart", &SportClient::Heart)
      .def("Pose", &SportClient::Pose)
      .def("Scrape", &SportClient::Scrape)
      .def("FrontFlip", &SportClient::FrontFlip)
      .def("FrontJump", &SportClient::FrontJump)
      .def("FrontPounce", &SportClient::FrontPounce)
      .def("Dance1", &SportClient::Dance1)
      .def("Dance2", &SportClient::Dance2)
      .def("Dance3", &SportClient::Dance3)
      .def("Dance4", &SportClient::Dance4)
      .def("HopSpinLeft", &SportClient::HopSpinLeft)
      .def("HopSpinRight", &SportClient::HopSpinRight)
      .def("WiggleHips", &SportClient::WiggleHips)
      .def("GetState", &SportClient::GetState)
      .def("EconomicGait", &SportClient::EconomicGait);
    // sport_client.hpp

  py::class_<ChannelFactory>(m, "ChannelFactory")
      .def("Instance", &ChannelFactory::Instance, py::return_value_policy::reference)
      .def("Init", py::overload_cast<int32_t, const std::string&>(&ChannelFactory::Init));
    //   .def("Init", py::overload_cast<std::string&>(&ChannelFactory::Init))
    //   .def("Init", py::overload_cast<unitree::common::JsonMap&>(&ChannelFactory::Init))
    //   .def("Release", &ChannelFactory::Release);

}
