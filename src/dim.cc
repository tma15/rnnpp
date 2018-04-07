#include "dim.h"

namespace rnnpp {

bool operator==(const Dim &lhs, const Dim &rhs) {
  if (lhs.shape.size() != rhs.shape.size()) {
    return false;
  }
  for (int i=0; i < lhs.shape.size(); ++i) {
    if (lhs.shape[i] != rhs.shape[i]) {
      return false;
    }
  }
  return true;
}


} // namespace rnnpp
