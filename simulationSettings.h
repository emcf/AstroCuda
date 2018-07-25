#pragma once
// Simulation dimensions
#define WIDTH 900
#define HEIGHT 900
// Maximum amoount of particles per bucket (leaf node) in octree
#define MAX_PARTICLES_PER_BUCKET 1024
// Amount of particles
#define N 8192
// Amount of octants to allocate for octreeSystem.octantList. Should be the maximum amount of possible octants for memory safety
// This algorithm shown is not 100% safe; it is merely a memory-conservative estimate on how many octs might be needed
// TODO: Find a good value for this
#define OCT_LIST_ALLOC (N*sizeof(octant)) / MAX_PARTICLES_PER_BUCKET
#define MASS_PER_KERNEL 10.0f
// Gravity settings
#define G_CONST 0.5f
#define GRAVITY_SOFTENING 0.000001f
#define DELTA_TIME 0.02f
// Display settings
#define DRAW_BUCKETS true
#define DRAW_MORTON_CURVE true
#define DRAW_PARTICLES true
