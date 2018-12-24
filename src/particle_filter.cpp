/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	num_particles = 10;
	default_random_engine gen;

	//Create Normal Gaussian distribution based on the passed in standard deviations.
	//Using the normal distribution particle are generated with random noise.
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for(int i = 0; i < num_particles; ++i )
	{
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		particles.push_back(p);
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	default_random_engine gen;

	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	//Compute predicted x,y and theta for all the particles and generate
	//predicted based on Gaussian distribtuion using passed in std
	for(int i = 0; i < num_particles; ++i )
	{

		if(0 >= fabs(yaw_rate))
		{
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		else
		{
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement(Nearest neighbor) and assign the
	// observed measurement to this particular landmark.

	for( auto& observedLm: observations)
	{
		//find the nearest predicted landmark for observed measurement
		double minPredDist = std::numeric_limits<double>::max();
		for( auto const& predictedLm: predicted)
		{
			double calcDist = dist(observedLm.x,observedLm.y,predictedLm.x, predictedLm.y);
			if(minPredDist > calcDist)
			{
				minPredDist = calcDist;
				observedLm.id = predictedLm.id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

	for( auto& p: particles)
	{
		//For each particle Observed landmark is in vehicle coordinates,
		//Need to convert into map coordinates using the formulas
		std::vector<LandmarkObs> mappedObservations;
		for(auto const& observedLm: observations)
		{
			LandmarkObs mappedLm;
			mappedLm.id = observedLm.id;

			//# transform to map x coordinate
			//x_map= x_part + (np.cos(theta) * x_obs) - (np.sin(theta) * y_obs)
			mappedLm.x = p.x + (( cos(p.theta) * observedLm.x) - (sin(p.theta) * observedLm.y));

			//# transform to map y coordinate
			//y_map= y_part + (np.sin(theta) * x_obs) + (np.cos(theta) * y_obs)
			mappedLm.y = p.y + (( sin(p.theta) * observedLm.x) + (cos(p.theta) * observedLm.y));
			mappedObservations.push_back(mappedLm);
		}

		//Filter maplandmarks within sensor range
		std::vector<LandmarkObs> predictions;
		for(auto const& landmark: map_landmarks.landmark_list)
		{
			double landmarkDist = dist(p.x, p.y, landmark.x_f, landmark.y_f);
			if( landmarkDist < sensor_range)
			{
				predictions.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
			}
		}
		dataAssociation(predictions,mappedObservations);

		//Compute multivariate Guassian probability
		p.weight = 1.0;

		for(auto const& observedLm: mappedObservations)
		{
			for(auto const& predictedLm: predictions)
			{
				if(predictedLm.id == observedLm.id)
				{
					//# calculate normalization term
				    //gauss_norm= (1/(2 * np.pi * sig_x * sig_y))
					double guass_norm = ( 1/(2 * M_PI * std_landmark[0] * std_landmark[1]));

					//# calculate exponent
					//exponent= ((x_obs - mu_x)**2)/(2 * sig_x**2) + ((y_obs - mu_y)**2)/(2 * sig_y**2)

					double exponent = (pow(observedLm.x - predictedLm.x, 2) / (2 * pow(std_landmark[0], 2)) +
									   pow(observedLm.y - predictedLm.y, 2) / (2 * pow(std_landmark[1], 2)));

					//# calculate weight using normalization terms and exponent
					//weight= gauss_norm * math.exp(-exponent)

					double weight =  guass_norm * exp(-exponent);
					p.weight *= weight;
				}
			}
		}
		weights.push_back(p.weight);
	}
}

void ParticleFilter::resample() {

	 // http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	 std::random_device rd;
	 std::mt19937 gen(rd());
	 std::discrete_distribution<> disc_dist(weights.begin(), weights.end());

	 //Create new sample particles using discrete distribution
	 std::vector<Particle> newParticles;
     newParticles.resize(particles.size());
	 for(std::vector<int>::size_type i = 0; i < particles.size(); i++) {
	     /* std::cout << v[i]; ... */
		 int new_idx = disc_dist(gen);
		 newParticles[i] = particles[new_idx];
	 }
	 particles = newParticles;
	 weights.clear();

}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
