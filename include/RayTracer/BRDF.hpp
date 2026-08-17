#ifndef BRDF_HPP
#define BRDF_HPP

#include <complex>
#include <cmath>
#include <memory>


class RadiationCharakteristics {
public:
	virtual double operator()(
		const double& cos_theta) = 0;

	virtual ~RadiationCharakteristics() = default;
};


class Schlick_Approximation: public RadiationCharakteristics {
public:
	Schlick_Approximation(const double& n1, const double& n2)
		:RadiationCharakteristics()
	{
		m_R0 = calc_R0(n1, n2);
	}

	Schlick_Approximation(const double& n1, const std::complex<double>& n2)
		:RadiationCharakteristics()
	{
		m_R0 = calc_R0(n1, n2);
	}

	double operator()(
		const double& cos_theta) override 
	{	
		//std::cout << "M_R0 " << m_R0 << std::endl;
		auto val = (m_R0 + (1 - m_R0) * std::pow(1.0 - cos_theta, 5));
		//std::cout << "BRDF Scaling " << val << std::endl;
		return val;
	}

	~Schlick_Approximation() override = default;

private:

	double calc_R0(const double& n1, const double& n2) noexcept {
		return std::pow((n1 - n2) / (n1 + n2), 2.0);
	}

	double calc_R0(const double& n1, const std::complex<double>& n2) noexcept {
		const double img_2 = n2.imag() * n2.imag();
		const double num = (n1 - n2.real());
		const double denom = (n1 + n2.real());
		return ((num * num) + img_2) / ((denom * denom) + img_2);
	}

	double m_R0{};
};

class Lambert : public RadiationCharakteristics {
public:
	Lambert()
		:RadiationCharakteristics(){ }

	~Lambert() override = default;

	double operator()(const double& cos_theta) {
		return std::max(cos_theta, 0.0);
	}

};

// Can be initialized with a custom Radiation Charakteristic function
// Defaults to Schlick Approximation
class BRDF {
public:
	BRDF(const double& n1, const double& n2)
		:m_characteristic{new Schlick_Approximation(n1, n2)}{ }

	BRDF(const double& n1, const std::complex<double>& n2)
		:m_characteristic{new Schlick_Approximation(n1, n2)} { }

	BRDF(std::unique_ptr<RadiationCharakteristics>&& radiation)
		: m_characteristic{std::move(radiation)} { }

	// cos_theta - Angle from display normal to sight ray
	double get_Reflection(const double& cos_theta) 
	{			
		// auto schlick = (*m_characteristic)(cos_theta);

		// auto distance = 1 / (dist * dist);

		// std::cout << "Schlickspprosimation " << schlick << '\n';

		// std::cout << "Distance " << distance << '\n';

		return (*m_characteristic)(cos_theta);
	}
	
private:
	std::unique_ptr<RadiationCharakteristics> m_characteristic{ nullptr };
};







#endif