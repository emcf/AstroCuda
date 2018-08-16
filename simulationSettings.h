#pragma once
// Simulation dimensions
#define WIDTH 920
#define HEIGHT 950
// Maximum amoount of particles per bucket (leaf node) in quadtree
// Must satisfy the following properties:
// MAX_PARTICLES_PER_BUCKET <= 1024
// MAX_PARTICLES_PER_BUCKET * sizeof(deviceParticle) <= 49152
#define MAX_PARTICLES_PER_BUCKET 512
// Amount of particles
#define N 2048 // 2^12 particles works well
#define PARTICLE_DATA_LENGTH 11
// Amount of quads to allocate for quadtreeSystem.quadList. Should be the maximum amount of possible quads for memory safety
// This algorithm shown is not 100% safe; it is merely a memory-conservative estimate on how many quads might be needed
// TODO: Find a good value for this
#define QUAD_LIST_ALLOC (N*sizeof(quad)) / MAX_PARTICLES_PER_BUCKET
// SPH settings
#define DELTA_TIME 0.1f
// Display settings
#define DRAW_BUCKETS true
#define DRAW_MORTON_CURVE true
#define DRAW_PARTICLES true
#define DRAW_SMOOTHING true
// Random particle distribution type
#define UNIFORM_DISTRIBUTION true
