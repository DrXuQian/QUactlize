// L246 -- host/constexpr closure for owned CuTe coordinate-iterator shapes.

#include <cstdio>
#include <type_traits>
#include <utility>

#include <cute/tensor.hpp>
#include <cute/ppu_stride.hpp>

namespace {

template <class Iterator>
constexpr bool owns_shape_v =
    !std::is_reference_v<decltype(std::declval<Iterator&>().shape)>;

constexpr bool forward_scalar_temporary() {
  auto iterator = cute::make_coord_iterator(5);
  static_assert(owns_shape_v<decltype(iterator)>);

  for (int expected = 0; expected < 5; ++expected) {
    if (*iterator != expected) return false;
    ++iterator;
  }
  return iterator == cute::ForwardCoordIteratorSentinel{};
}

constexpr bool forward_nested_temporary_and_lvalue_agree() {
  auto shape = cute::make_shape(cute::make_shape(2, 3), 4);
  auto from_lvalue = cute::make_coord_iterator(shape);
  auto from_temporary =
      cute::make_coord_iterator(cute::make_shape(cute::make_shape(2, 3), 4));
  static_assert(owns_shape_v<decltype(from_lvalue)>);
  static_assert(owns_shape_v<decltype(from_temporary)>);

  for (int expected = 0; expected < 24; ++expected) {
    if (*from_lvalue != *from_temporary) return false;
    if (cute::crd2idx(*from_temporary, shape) != expected) return false;
    ++from_lvalue;
    ++from_temporary;
  }
  return from_lvalue == cute::ForwardCoordIteratorSentinel{} &&
         from_temporary == cute::ForwardCoordIteratorSentinel{};
}

constexpr bool splitk_scalar_temporary() {
  auto iterator = cute::make_splitk_coord_iterator(11, 2, 3);
  static_assert(owns_shape_v<decltype(iterator)>);

  for (int expected : {2, 5, 8}) {
    if (*iterator != expected) return false;
    ++iterator;
  }
  return iterator == cute::SplitkCoordIteratorSentinal{};
}

constexpr bool splitk_nested_temporary_and_lvalue_agree() {
  auto shape = cute::make_shape(cute::make_shape(2, 3), 4);
  auto from_lvalue = cute::make_splitk_coord_iterator(shape, 2, 5);
  auto from_temporary = cute::make_splitk_coord_iterator(
      cute::make_shape(cute::make_shape(2, 3), 4), 2, 5);
  static_assert(owns_shape_v<decltype(from_lvalue)>);
  static_assert(owns_shape_v<decltype(from_temporary)>);

  for (int expected : {2, 7, 12, 17, 22}) {
    if (*from_lvalue != *from_temporary) return false;
    if (cute::crd2idx(*from_temporary, shape) != expected) return false;
    ++from_lvalue;
    ++from_temporary;
  }
  return from_lvalue == cute::SplitkCoordIteratorSentinal{} &&
         from_temporary == cute::SplitkCoordIteratorSentinal{};
}

}  // namespace

int main() {
  bool const pass =
      forward_scalar_temporary() &&
      forward_nested_temporary_and_lvalue_agree() &&
      splitk_scalar_temporary() &&
      splitk_nested_temporary_and_lvalue_agree();
  std::printf(
      "L246 COORD_ITERATOR_LIFETIME %s forward=scalar+nested "
      "splitk=scalar+nested lvalue=equivalent ownership=value\n",
      pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
