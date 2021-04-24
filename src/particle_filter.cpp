/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 * 
 * Modified on: Apr 24, 2021
 * Author: Choonggeun Song
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

static std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to 
   * first position (based on estimates of x, y, theta and their uncertainties
   * from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   * Consult particle_filter.h for more information about this method 
   * (and others in this file).
   */
  num_particles = 100;  // Set the number of particles

  std::normal_distribution<double> N_x(x, std[0]);
  std::normal_distribution<double> N_y(y, std[1]);
  std::normal_distribution<double> N_theta(theta, std[2]);

  for (int id = 0; id < num_particles; id++) {
      Particle particle;
      particle.id = id;
      particle.x = N_x(gen);
      particle.y = N_y(gen);
      particle.theta = N_theta(gen);
      particle.weight = 1;

      particles.push_back(particle);
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  for (int idx = 0; idx < num_particles; idx++) {
    
    double new_x = 0;
    double new_y = 0;
    double new_theta = 0;

    if (abs(yaw_rate) < 1e-5) {
      new_theta = particles[idx].theta;
      new_x = particles[idx].x + velocity*delta_t*cos(particles[idx].theta);
      new_y = particles[idx].y + velocity*delta_t*sin(particles[idx].theta);
    } else {
      new_theta = particles[idx].theta + yaw_rate*delta_t;
      new_x = particles[idx].x + velocity/yaw_rate*(sin(new_theta) - sin(particles[idx].theta));
      new_y = particles[idx].y + velocity/yaw_rate*(cos(particles[idx].theta) - cos(new_theta));
    }    

    std::normal_distribution<double> N_x(new_x, std_pos[0]);
    std::normal_distribution<double> N_y(new_y, std_pos[1]);
    std::normal_distribution<double> N_theta(new_theta, std_pos[2]);

    particles[idx].x = N_x(gen);
    particles[idx].y = N_y(gen);
    particles[idx].theta = N_theta(gen);
  } 
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * Find the predicted measurement that is closest to each 
   * observed measurement and assign the observed measurement to this 
   * particular landmark.
   * this method will NOT be called by the grading code. But you will 
   * probably find it useful to implement this method and use it as a helper 
   * during the updateWeights phase.
   */
  for (LandmarkObs& obs : observations) {
      vector<double> results;      
      for (LandmarkObs& pred : predicted)
         results.push_back(dist(obs.x, obs.y, pred.x, pred.y));
         
      int minElemIdx = std::min_element(results.begin(), results.end()) - results.begin();
      obs.id = predicted[minElemIdx].id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * Update the weights of each particle using a mult-variate Gaussian 
   * distribution. You can read more about this distribution here: 
   * https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * The observations are given in the VEHICLE'S coordinate system. 
   * Your particles are located according to the MAP'S coordinate system. 
   * You will need to transform between the two systems. Keep in mind that
   * this transformation requires both rotation AND translation (but no scaling).
   * The following is a good resource for the theory:
   * https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   * and the following is a good resource for the actual equation to implement
   * (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  // Check each particle
  for (int iter = 0; iter < num_particles; iter++) {

    // Get x, y, and theta of the current particle
    double veh_x = particles[iter].x;
    double veh_y = particles[iter].y;
    double veh_theta = particles[iter].theta;

    // Predicted landmark vector within the sensor range of this current particle
    vector<LandmarkObs> predicted_landmarks;

    for (const Map::single_landmark_s& map_landmark : map_landmarks.landmark_list) {

      // The distance between the current (x,y) of the vehicle and the landmark (x,y) in this map.
      double result = dist(veh_x, veh_y, static_cast<double>(map_landmark.x_f), static_cast<double>(map_landmark.y_f));
      // This sensor range is the circle.
      if (result < sensor_range) {
        LandmarkObs predicted_landmark;
        predicted_landmark.id = map_landmark.id_i;
        predicted_landmark.x = static_cast<double>(map_landmark.x_f);
        predicted_landmark.y = static_cast<double>(map_landmark.y_f);
        predicted_landmarks.push_back(predicted_landmark);
      }
    }

    // Transformed observation markers from the vehicle's coordinates to the map's coordinates, with respect to our particle.
    vector<LandmarkObs> transformed_observations;

    for (unsigned int obs_idx = 0; obs_idx < observations.size(); obs_idx++) {

      // Convert observation from particle(vehicle) to map coordinate system
      LandmarkObs map_obs;
      map_obs.id = observations[obs_idx].id;
      map_obs.x = cos(veh_theta)*observations[obs_idx].x - sin(veh_theta)*observations[obs_idx].y + veh_x;
      map_obs.y = sin(veh_theta)*observations[obs_idx].x + cos(veh_theta)*observations[obs_idx].y + veh_y;

      transformed_observations.push_back(map_obs);
    }

    // Conduct nearest neighbor data association to match transformed observations to predicted landmarks.
    dataAssociation(predicted_landmarks, transformed_observations);

    // Calculate the Particle's Final Weight
    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];

    // calculate normalization term
    double gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

    // initialize the particle's weight before updating.
    particles[iter].weight = 1;

    for (const LandmarkObs& trans_obs : transformed_observations) {

      double x_obs = trans_obs.x; 
      double y_obs = trans_obs.y;
      double mu_x = 0;
      double mu_y = 0;
      
      for (const LandmarkObs& landmark: predicted_landmarks)
        if (trans_obs.id == landmark.id) {
            mu_x = landmark.x;
            mu_y = landmark.y;
        }  

      // calculate exponent
      double exponent = (std::pow(x_obs - mu_x, 2) / (2 * std::pow(sig_x, 2))) + (std::pow(y_obs - mu_y, 2) / (2 * std::pow(sig_y, 2)));
    
      // calculate weight using normalization terms and exponent
      double trans_obs_weight = gauss_norm * exp(-exponent);

      // multiply this calcualted weight to this particle's weight to update
      particles[iter].weight *= trans_obs_weight;
    }
  } 
}

void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional to their weight. 
   * You may find std::discrete_distribution helpful here.
   * http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  // all of the current weights
  vector<double> current_weights;
  for (const Particle& particle : particles)
      current_weights.push_back(particle.weight);

  // Applying resampling wheel!
  std::uniform_int_distribution<int> index_uniform(0, num_particles-1);
  auto particle_idx = index_uniform(gen);

  double beta = 0.0;
  double max_cur_weight = *std::max_element(current_weights.begin(), current_weights.end());
  std::uniform_real_distribution<double> weight_uniform(0, max_cur_weight);
  auto weight_rdn_gen = weight_uniform(gen);

  vector<Particle> resampled_particles;
  for (int p_iter = 0; p_iter < num_particles; p_iter++) {
      beta += weight_rdn_gen*2.0*max_cur_weight;
      while (beta > current_weights[particle_idx]) {
        beta -= current_weights[particle_idx];
        particle_idx = (particle_idx + 1) % num_particles;
      }
      resampled_particles.push_back(particles[particle_idx]);
  }

  particles = resampled_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}