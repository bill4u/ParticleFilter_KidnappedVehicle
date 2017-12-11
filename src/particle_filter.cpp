/*
 * particle_filter.cpp
 *
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

#define Num_Particles 100
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;
	if (is_initialized) {
		return;
	}
	
	num_particles = Num_Particles;
	
	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	normal_distribution<double> distance_x(x, std_x);
	normal_distribution<double> distance_y(y, std_y);
	normal_distribution<double> distance_theta(theta, std_theta);
	
	for (int i = 0; i <num_particles; i++)
	{
		Particle particle;
		particle.id = i;
		particle.x = distance_x(gen);
		particle.y = distance_y(gen);
		particle.theta = distance_theta(gen);
		particle.weight = 1.0;
		
		particles.push_back(particle);
		
	}

	is_initialized = true;
	
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	
	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];
	
	normal_distribution<double> distance_x(0,std_x);
	normal_distribution<double> distance_y(0,std_y);
	normal_distribution<double> distance_theta(0,std_theta);
	
	for (int i = 0; i < num_particles; i++) {
		double theta = particles[i].theta;
		
		
		
		if ( fabs(yaw_rate) < 0.00001) // yaw does notchange
		{
			particles[i].x += velocity * delta_t * cos (theta);
			particles[i].y += velocity * delta_t * sin (theta);
		}
		else {
			/*
			x​f​​ =x​0​​ +​theta˙​​ ​​v​​ [sin(theta0​​ +​theta˙​​ (dt))−sin(theta0​​ )]
			y​f​​ =y​0​​ +​​theta˙​​ ​​v​​ [cos(theta0​​ )−cos(theta0​​ +​theta˙​​ (dt))]
			thetaf​​ =theta0​​ +​theta˙​​ (dt)
			*/
			particles[i].x += velocity / yaw_rate * ( sin (theta + yaw_rate * delta_t ) - sin( theta ) );
			particles[i].y += velocity / yaw_rate * ( cos (theta) - cos( theta + yaw_rate * delta_t ) );
			particles[i].theta += yaw_rate * delta_t;
		}
		//Add random Gaussian Noise
		particles[i].x += distance_x(gen);
		particles[i].y += distance_y(gen);
		particles[i].theta += distance_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	int noObs = observations.size();
	int noPreds = predicted.size();
	
	for (int i = 0; i < noObs; i++) {
		double minDistance = 1000000;
		int mapID = -1;
		
		for (int j = 0; j < noPreds; j++) {
		double x_dist = observations[i].x - predicted[j].x;
		double y_dist = observations[i].y - predicted[j].y;
		
		double dist = (x_dist * x_dist + y_dist * y_dist);
		
		if (dist < minDistance) {
			minDistance = dist;
			mapID = predicted[j].id;
			}
		}
		observations[i].id = mapID;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html


	double stdLandmarkRange = std_landmark[0];
	double stdLandmarkBearing = std_landmark[1];
	
	for (int i = 0; i < num_particles; i ++){
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;
		
		double sensor_range_2 = sensor_range * sensor_range;
		vector <LandmarkObs> inRangeLandmarks;
		//Filter landmarks and particles
		for (int j =0; j < map_landmarks.landmark_list.size(); j++) {
			float landmarkX = map_landmarks.landmark_list[j].x_f;
			float landmarkY = map_landmarks.landmark_list[j].y_f;
			int id = map_landmarks.landmark_list[j].id_i;
			double dX = x - landmarkX;
			double dY = y - landmarkY;
			if (dX* dX + dY* dY <= sensor_range_2) {
				inRangeLandmarks.push_back(LandmarkObs{id, landmarkX, landmarkY });
			}
		}
		vector<LandmarkObs> mappedObservations;
		for (int j = 0; j< observations.size(); j++){
			double xx = cos(theta)*observations[j].x - sin(theta)*observations[j].y + x;
			double yy = sin(theta)*observations[j].x + cos(theta)*observations[j].y + y;
			mappedObservations.push_back(LandmarkObs{ observations[j].id, xx, yy });
		}
		dataAssociation(inRangeLandmarks, mappedObservations);

	particles[i].weight=1.0;
    // Calculate weights.
    for(int j = 0; j < mappedObservations.size(); j++) {
      //Get Obxy and IDs
	  double observationX = mappedObservations[j].x;
      double observationY = mappedObservations[j].y;
	  int landmarkId = mappedObservations[j].id;
	  double landmarkX, landmarkY;
      int k = 0;
      int nLandmarks = inRangeLandmarks.size();
      bool found = false;
      while( !found && k < nLandmarks ) {
        if ( inRangeLandmarks[k].id == landmarkId) {
          found = true;
          landmarkX = inRangeLandmarks[k].x;
          landmarkY = inRangeLandmarks[k].y;
        }
        k++;
      }

      double dX = observationX - landmarkX;
      double dY = observationY - landmarkY;
	  double Normalizer = 2*stdLandmarkBearing*stdLandmarkRange;
	  double dist_temp = dX*dX/(Normalizer) + dY*dY/(Normalizer);
	  //Gaussian Dist
      double weight = ( 1/( Normalizer*M_PI	)) * exp( -(( dist_temp) ));
	  
      if (weight == 0) {
        particles[i].weight *= 0.00001;
      } else {
        particles[i].weight *= weight;
      }
    }
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	 vector<double> weights;
	double maxWeight = 0;
	for(int i = 0; i < num_particles; i++) {
		weights.push_back(particles[i].weight);
		if ( particles[i].weight > maxWeight ) {
		  maxWeight = particles[i].weight;
		}
	}

	// Samplers.
	//Beta Dist Sampler
	uniform_real_distribution<double> distDouble(0.0, maxWeight);
	// Index Dist Sampler
	uniform_int_distribution<int> distInt(0, num_particles - 1);

	// Generating index.
	int index = distInt(gen);

	double beta = 0.0;

	vector<Particle> resampledParticles;
	for(int i = 0; i < num_particles; i++) {
	beta += distDouble(gen) * 2.0  * maxWeight;  //Make value larger than twice the max weight
	while( weights[index] < beta) {
	  beta -= weights[index];
	  index = (index + 1) % num_particles;
	}
	resampledParticles.push_back(particles[index]);
	}

	particles = resampledParticles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    // particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
	return particle;
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
