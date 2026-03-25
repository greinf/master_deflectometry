#ifndef ENUMS_H
#define ENUMS_H

enum class Shift_mode {
	four_phase_shift,
	user_defined,
	GrayValues,
};

enum class DisplayMode {
	UserInput,
	Automatic,
	max_value
};

enum class UnwrapMode {
	manually,
	manually_reference,
	opencv,
	max_value
};

enum class AcquisitionMode {
	UserInput,
	Automatic,
};

enum class ReferenceMode {
	cross,
	checkerboard
};

enum class Warping {
	homography,
	raycasting
};

enum class Rotation {
	eulerxy,
	rodrigeuz
};

#endif // !1
