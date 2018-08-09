#pragma once
// Simulation dimensions
#define WIDTH 900
#define HEIGHT 900
// Maximum amoount of particles per bucket (leaf node) in octree
#define MAX_PARTICLES_PER_BUCKET 1024
// Amount of particles
#define N 4096
#define PARTICLE_DATA_LENGTH 12
// Amount of octants to allocate for octreeSystem.octantList. Should be the maximum amount of possible octants for memory safety
// This algorithm shown is not 100% safe; it is merely a memory-conservative estimate on how many octs might be needed
// TODO: Find a good value for this
#define OCT_LIST_ALLOC (N*sizeof(octant)) / MAX_PARTICLES_PER_BUCKET
// Gravity settings
#define G_CONST 0.5f
#define DELTA_TIME 0.3f
// Display settings
#define DRAW_BUCKETS true
#define DRAW_MORTON_CURVE false
#define DRAW_PARTICLES true
// Random particle distribution type
#define UNIFORM_DISTRIBUTION true
