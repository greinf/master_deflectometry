#ifndef DEFLECTOMETRYUTILS_HPP
#define DEFLECTOMETRYUTILS_HPP

namespace _defl_ {
	namespace GrayCal {
		enum class Method {
			None,
			ActiveLut,
			PassiveLut,
			ActiveModel,
			PassiveModel,
			ActiveModel_Bias,
			PassiveModel_Bias,
			LocLutActive,
			LocLutPassive
		};
	};
}
 
#endif