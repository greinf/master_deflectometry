#ifndef ROWPOLICY_HPP
#define ROWPOLICY_HPP
#include <type_traits>

struct PerElement {};
struct UniformRowsCols {};

namespace policy_check {
	// generic: T is one of Ts...
	template<typename T, typename... Ts>
	struct is_one_of : std::bool_constant<(std::is_same_v<T, Ts> || ...)> {};

	template<typename T, typename... Ts>
	inline constexpr bool is_one_of_v = is_one_of<T, Ts...>::value;
}

// your specific predicate
template<typename T>
inline constexpr bool is_row_policy_v =
policy_check::is_one_of_v<std::decay_t<T>, PerElement, UniformRowsCols>;

#endif