#ifndef RNNPP_ERROR_H
#define RNNPP_ERROR_H

#include <sstream>

namespace rnnpp {

#define RNNPP_CHECK(cond, msg) do {           \
  if (!(cond)) {                              \
    std::stringstream ss;                     \
    ss << __FILE__ << " (" << __LINE__ << ")";\
    ss << " " << #cond << " " << msg;         \
    throw std::runtime_error(ss.str()); }     \
} while (0); 

} // namespace rnnpp

#endif // RNNPP_ERROR_H
