#ifndef BRDF_HPP
#define BRDF_HPP

#include <complex>
#include <cmath>
#include <memory>


class RadiationCharakteristics {
public:
	virtual double operator()(
		const double& cos_theta) = 0;
};


class Schlick_Approximation: public RadiationCharakteristics {
public:
	Schlick_Approximation(const double& n1, const double& n2)
	{
		m_R0 = calc_R0(n1, n2);
	}

	Schlick_Approximation(const double& n1, const std::complex<double>& n2)
	{
		m_R0 = calc_R0(n1, n2);
	}

	double operator()(
		const double& cos_theta) override 
	{
		return (m_R0 + (1 - m_R0) * std::pow(1.0 - cos_theta, 5));
	}

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

	double get_Reflection(const double& cos_theta) 
	{
		return (*m_characteristic)(cos_theta);
	}
	
private:
	std::unique_ptr<RadiationCharakteristics> m_characteristic{ nullptr };
};







#endif