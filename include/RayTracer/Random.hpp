#ifndef RANDOM_Ray_HPP
#define RANDOM_Ray_HPP

#include <random>
#include <thread>

struct RandomGenerator {
	[[nodiscard]] virtual float operator()() = 0;
	
	virtual ~RandomGenerator() = default;
};

struct MersenneTwister: public RandomGenerator {
	MersenneTwister():
		RandomGenerator() { }

	~MersenneTwister() override = default;

	// Creates Random Values [0,1]
	// Most efficient when called via multiple threads 
	[[nodiscard]] float operator()() {
		thread_local std::mt19937_64 generator(std::random_device{}());
		thread_local std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
		return distribution(generator);
	}
	



	/*[[nodiscard]] float generate() {
		thread_local std::mt19937_64 generator(std::random_device{}());
		thread_local std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
		return distribution(generator);
	}*/
};

#endif // "RANDOM_Ray_HPP"