# Builds this C++ project requiring OpenGL and CUDA using NVidia's NVCC compiler
nvcc *.cu -o build -lglut -lGLU -lGL -run 
