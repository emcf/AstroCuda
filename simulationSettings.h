#pragma once
// Simulation dimensions
#define WIDTH 1000
#define HEIGHT 1000
// Maximum amount of particles per bucket (leaf node) in quadtree
// Must satisfy the following properties:
// MAX_PARTICLES_PER_BUCKET <= 1024
// MAX_PARTICLES_PER_BUCKET * sizeof(deviceParticle) <= 49152
#define MAX_PARTICLES_PER_BUCKET 512
// Amount of particles
#define N 4096
#define PARTICLE_DATA_LENGTH 11
// Amount of quads to allocate for quadtreeSystem.quadList.
// This algorithm shown is not 100% safe; it is merely a memory-conservative estimate on how many quads might be needed
#define QUAD_LIST_ALLOC (N*sizeof(quad)) / MAX_PARTICLES_PER_BUCKET
// SPH settings
#define DELTA_TIME 0.2f
// Display settings
#define DRAW_BUCKETS true
#define DRAW_MORTON_CURVE true
#define DRAW_PARTICLES true
#define DRAW_SMOOTHING true
// Random particle distribution type.
// Be careful, since low simulation dimensions or high N values
// will result in floating point errors that crash the simulation
#define UNIFORM_DISTRIBUTION true
