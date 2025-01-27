/****************************************************************

  Generated by Eclipse Cyclone DDS IDL to CXX Translator
  File name: HelloWorldData.idl
  Source: HelloWorldData.hpp
  Cyclone DDS: v0.10.2

*****************************************************************/
#ifndef DDSCXX_HELLOWORLDDATA_HPP
#define DDSCXX_HELLOWORLDDATA_HPP

#include <cstdint>
#include <string>

namespace HelloWorldData
{
class Msg
{
private:
 int64_t userID_ = 0;
 std::string message_;

public:
  Msg() = default;

  explicit Msg(
    int64_t userID,
    const std::string& message) :
    userID_(userID),
    message_(message) { }

  int64_t userID() const { return this->userID_; }
  int64_t& userID() { return this->userID_; }
  void userID(int64_t _val_) { this->userID_ = _val_; }
  const std::string& message() const { return this->message_; }
  std::string& message() { return this->message_; }
  void message(const std::string& _val_) { this->message_ = _val_; }
  void message(std::string&& _val_) { this->message_ = _val_; }

  bool operator==(const Msg& _other) const
  {
    (void) _other;
    return userID_ == _other.userID_ &&
      message_ == _other.message_;
  }

  bool operator!=(const Msg& _other) const
  {
    return !(*this == _other);
  }

};

}

#include "dds/topic/TopicTraits.hpp"
#include "org/eclipse/cyclonedds/topic/datatopic.hpp"

namespace org {
namespace eclipse {
namespace cyclonedds {
namespace topic {

template <> constexpr const char* TopicTraits<::HelloWorldData::Msg>::getTypeName()
{
  return "HelloWorldData::Msg";
}

template <> constexpr bool TopicTraits<::HelloWorldData::Msg>::isSelfContained()
{
  return false;
}

template <> constexpr bool TopicTraits<::HelloWorldData::Msg>::isKeyless()
{
  return true;
}

#ifdef DDSCXX_HAS_TYPE_DISCOVERY
template<> constexpr unsigned int TopicTraits<::HelloWorldData::Msg>::type_map_blob_sz() { return 246; }
template<> constexpr unsigned int TopicTraits<::HelloWorldData::Msg>::type_info_blob_sz() { return 100; }
template<> inline const uint8_t * TopicTraits<::HelloWorldData::Msg>::type_map_blob() {
  static const uint8_t blob[] = {
 0x4c,  0x00,  0x00,  0x00,  0x01,  0x00,  0x00,  0x00,  0xf1,  0xc6,  0x4e,  0xc8,  0x69,  0x45,  0x24,  0x37, 
 0x97,  0x71,  0x0e,  0x16,  0xea,  0x09,  0x7f,  0x00,  0x34,  0x00,  0x00,  0x00,  0xf1,  0x51,  0x01,  0x00, 
 0x01,  0x00,  0x00,  0x00,  0x00,  0x00,  0x00,  0x00,  0x24,  0x00,  0x00,  0x00,  0x02,  0x00,  0x00,  0x00, 
 0x0b,  0x00,  0x00,  0x00,  0x00,  0x00,  0x00,  0x00,  0x01,  0x00,  0x05,  0x58,  0x5c,  0x95,  0x70,  0x00, 
 0x0c,  0x00,  0x00,  0x00,  0x01,  0x00,  0x00,  0x00,  0x01,  0x00,  0x70,  0x00,  0x78,  0xe7,  0x31,  0x02, 
 0x7a,  0x00,  0x00,  0x00,  0x01,  0x00,  0x00,  0x00,  0xf2,  0x70,  0xc8,  0x44,  0xb4,  0xff,  0x08,  0x3b, 
 0x24,  0x1e,  0x92,  0xa7,  0x08,  0x93,  0x7e,  0x00,  0x62,  0x00,  0x00,  0x00,  0xf2,  0x51,  0x01,  0x00, 
 0x1c,  0x00,  0x00,  0x00,  0x00,  0x00,  0x00,  0x00,  0x14,  0x00,  0x00,  0x00,  0x48,  0x65,  0x6c,  0x6c, 
 0x6f,  0x57,  0x6f,  0x72,  0x6c,  0x64,  0x44,  0x61,  0x74,  0x61,  0x3a,  0x3a,  0x4d,  0x73,  0x67,  0x00, 
 0x3a,  0x00,  0x00,  0x00,  0x02,  0x00,  0x00,  0x00,  0x15,  0x00,  0x00,  0x00,  0x00,  0x00,  0x00,  0x00, 
 0x01,  0x00,  0x05,  0x00,  0x07,  0x00,  0x00,  0x00,  0x75,  0x73,  0x65,  0x72,  0x49,  0x44,  0x00,  0x00, 
 0x00,  0x00,  0x00,  0x00,  0x16,  0x00,  0x00,  0x00,  0x01,  0x00,  0x00,  0x00,  0x01,  0x00,  0x70,  0x00, 
 0x08,  0x00,  0x00,  0x00,  0x6d,  0x65,  0x73,  0x73,  0x61,  0x67,  0x65,  0x00,  0x00,  0x00,  0x00,  0x00, 
 0x22,  0x00,  0x00,  0x00,  0x01,  0x00,  0x00,  0x00,  0xf2,  0x70,  0xc8,  0x44,  0xb4,  0xff,  0x08,  0x3b, 
 0x24,  0x1e,  0x92,  0xa7,  0x08,  0x93,  0x7e,  0xf1,  0xc6,  0x4e,  0xc8,  0x69,  0x45,  0x24,  0x37,  0x97, 
 0x71,  0x0e,  0x16,  0xea,  0x09,  0x7f, };
  return blob;
}
template<> inline const uint8_t * TopicTraits<::HelloWorldData::Msg>::type_info_blob() {
  static const uint8_t blob[] = {
 0x60,  0x00,  0x00,  0x00,  0x01,  0x10,  0x00,  0x40,  0x28,  0x00,  0x00,  0x00,  0x24,  0x00,  0x00,  0x00, 
 0x14,  0x00,  0x00,  0x00,  0xf1,  0xc6,  0x4e,  0xc8,  0x69,  0x45,  0x24,  0x37,  0x97,  0x71,  0x0e,  0x16, 
 0xea,  0x09,  0x7f,  0x00,  0x38,  0x00,  0x00,  0x00,  0x00,  0x00,  0x00,  0x00,  0x04,  0x00,  0x00,  0x00, 
 0x00,  0x00,  0x00,  0x00,  0x02,  0x10,  0x00,  0x40,  0x28,  0x00,  0x00,  0x00,  0x24,  0x00,  0x00,  0x00, 
 0x14,  0x00,  0x00,  0x00,  0xf2,  0x70,  0xc8,  0x44,  0xb4,  0xff,  0x08,  0x3b,  0x24,  0x1e,  0x92,  0xa7, 
 0x08,  0x93,  0x7e,  0x00,  0x66,  0x00,  0x00,  0x00,  0x00,  0x00,  0x00,  0x00,  0x04,  0x00,  0x00,  0x00, 
 0x00,  0x00,  0x00,  0x00, };
  return blob;
}
#endif //DDSCXX_HAS_TYPE_DISCOVERY

} //namespace topic
} //namespace cyclonedds
} //namespace eclipse
} //namespace org

namespace dds {
namespace topic {

template <>
struct topic_type_name<::HelloWorldData::Msg>
{
    static std::string value()
    {
      return org::eclipse::cyclonedds::topic::TopicTraits<::HelloWorldData::Msg>::getTypeName();
    }
};

}
}

REGISTER_TOPIC_TYPE(::HelloWorldData::Msg)

namespace org{
namespace eclipse{
namespace cyclonedds{
namespace core{
namespace cdr{

template<>
propvec &get_type_props<::HelloWorldData::Msg>();

template<typename T, std::enable_if_t<std::is_base_of<cdr_stream, T>::value, bool> = true >
bool write(T& streamer, const ::HelloWorldData::Msg& instance, entity_properties_t *props) {
  (void)instance;
  if (!streamer.start_struct(*props))
    return false;
  auto prop = streamer.first_entity(props);
  while (prop) {
    switch (prop->m_id) {
      case 0:
      if (!streamer.start_member(*prop))
        return false;
      if (!write(streamer, instance.userID()))
        return false;
      if (!streamer.finish_member(*prop))
        return false;
      break;
      case 1:
      if (!streamer.start_member(*prop))
        return false;
      if (!write_string(streamer, instance.message(), 0))
        return false;
      if (!streamer.finish_member(*prop))
        return false;
      break;
    }
    prop = streamer.next_entity(prop);
  }
  return streamer.finish_struct(*props);
}

template<typename S, std::enable_if_t<std::is_base_of<cdr_stream, S>::value, bool> = true >
bool write(S& str, const ::HelloWorldData::Msg& instance, bool as_key) {
  auto &props = get_type_props<::HelloWorldData::Msg>();
  str.set_mode(cdr_stream::stream_mode::write, as_key);
  return write(str, instance, props.data()); 
}

template<typename T, std::enable_if_t<std::is_base_of<cdr_stream, T>::value, bool> = true >
bool read(T& streamer, ::HelloWorldData::Msg& instance, entity_properties_t *props) {
  (void)instance;
  if (!streamer.start_struct(*props))
    return false;
  auto prop = streamer.first_entity(props);
  while (prop) {
    switch (prop->m_id) {
      case 0:
      if (!streamer.start_member(*prop))
        return false;
      if (!read(streamer, instance.userID()))
        return false;
      if (!streamer.finish_member(*prop))
        return false;
      break;
      case 1:
      if (!streamer.start_member(*prop))
        return false;
      if (!read_string(streamer, instance.message(), 0))
        return false;
      if (!streamer.finish_member(*prop))
        return false;
      break;
    }
    prop = streamer.next_entity(prop);
  }
  return streamer.finish_struct(*props);
}

template<typename S, std::enable_if_t<std::is_base_of<cdr_stream, S>::value, bool> = true >
bool read(S& str, ::HelloWorldData::Msg& instance, bool as_key) {
  auto &props = get_type_props<::HelloWorldData::Msg>();
  str.set_mode(cdr_stream::stream_mode::read, as_key);
  return read(str, instance, props.data()); 
}

template<typename T, std::enable_if_t<std::is_base_of<cdr_stream, T>::value, bool> = true >
bool move(T& streamer, const ::HelloWorldData::Msg& instance, entity_properties_t *props) {
  (void)instance;
  if (!streamer.start_struct(*props))
    return false;
  auto prop = streamer.first_entity(props);
  while (prop) {
    switch (prop->m_id) {
      case 0:
      if (!streamer.start_member(*prop))
        return false;
      if (!move(streamer, instance.userID()))
        return false;
      if (!streamer.finish_member(*prop))
        return false;
      break;
      case 1:
      if (!streamer.start_member(*prop))
        return false;
      if (!move_string(streamer, instance.message(), 0))
        return false;
      if (!streamer.finish_member(*prop))
        return false;
      break;
    }
    prop = streamer.next_entity(prop);
  }
  return streamer.finish_struct(*props);
}

template<typename S, std::enable_if_t<std::is_base_of<cdr_stream, S>::value, bool> = true >
bool move(S& str, const ::HelloWorldData::Msg& instance, bool as_key) {
  auto &props = get_type_props<::HelloWorldData::Msg>();
  str.set_mode(cdr_stream::stream_mode::move, as_key);
  return move(str, instance, props.data()); 
}

template<typename T, std::enable_if_t<std::is_base_of<cdr_stream, T>::value, bool> = true >
bool max(T& streamer, const ::HelloWorldData::Msg& instance, entity_properties_t *props) {
  (void)instance;
  if (!streamer.start_struct(*props))
    return false;
  auto prop = streamer.first_entity(props);
  while (prop) {
    switch (prop->m_id) {
      case 0:
      if (!streamer.start_member(*prop))
        return false;
      if (!max(streamer, instance.userID()))
        return false;
      if (!streamer.finish_member(*prop))
        return false;
      break;
      case 1:
      if (!streamer.start_member(*prop))
        return false;
      if (!max_string(streamer, instance.message(), 0))
        return false;
      if (!streamer.finish_member(*prop))
        return false;
      break;
    }
    prop = streamer.next_entity(prop);
  }
  return streamer.finish_struct(*props);
}

template<typename S, std::enable_if_t<std::is_base_of<cdr_stream, S>::value, bool> = true >
bool max(S& str, const ::HelloWorldData::Msg& instance, bool as_key) {
  auto &props = get_type_props<::HelloWorldData::Msg>();
  str.set_mode(cdr_stream::stream_mode::max, as_key);
  return max(str, instance, props.data()); 
}

} //namespace cdr
} //namespace core
} //namespace cyclonedds
} //namespace eclipse
} //namespace org

#endif // DDSCXX_HELLOWORLDDATA_HPP
