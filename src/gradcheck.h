#ifndef RNNPP_GRADCHECK_H_
#define RNNPP_GRADCHECK_H_

#include "expr.h"

namespace rnnpp {

bool gradient_check(Expression &expr);

} // namespace rnnpp

#endif // RNNPP_GRADCHECK_H_
