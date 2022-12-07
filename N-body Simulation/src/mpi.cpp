#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <mpi.h>
#include <iostream>
#include <string.h>

#ifdef GUI
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "./headers/physics.h"
#include "./headers/logger.h"


int n_body;
int n_iteration;


int my_rank;
int world_size;

std::chrono::duration<double> time_span_total;

void generate_data(double *m, double *x,double *y,double *vx,double *vy, int n, int n_padded) {
    // Generate proper initial position and mass for better visualization
    srand((unsigned)time(NULL));
    for (int i = 0; i < n_padded; i++) {
        if (i < n) {
            m[i] = rand() % max_mass + 1.0f;
            x[i] = 2000.0f + rand() % (bound_x / 4);
            y[i] = 2000.0f + rand() % (bound_y / 4);
            vx[i] = 0.0f;
            vy[i] = 0.0f;
        }
        else {
            m[i] = 0;
            x[i] = 0;
            y[i] = 0;
            vx[i] = 0;
            vy[i] = 0;
        }
    }
}


void update_position(double *m, double *x, double *y, double *vx, double *vy, int n) {
    // update position 
    for(int i = 0; i < n; i++){
        x[i] += vx[i] * dt;
        y[i] += vy[i] * dt;
    }
}


void update_velocity(double *m, double *m_total, double *x, double *y, double *x_total, double *y_total, double *vx, double *vy, int n) {
    // calculate force and acceleration, update velocity
    double f_x;
    double f_y;
    double _f;
    double a_x;
    double a_y;
    double d_x;
    double d_y;
    double d_square;
    
    for(int i = 0; i < n; i++){
        f_x = 0;
        f_y = 0;
        // calculate the distance and then accumlate the forces and update velocity if collided
        for(int j = 0; j < n; j++){
            if(i == j) break;        
            d_x = x[j] - x[i];
            d_y = y[j] - y[i];
            d_square = pow(d_x , 2) + pow(d_y , 2);
            if (radius2 < d_square)
            //no collision
            { 
                _f = (gravity_const * m[i] * m[j]) / (d_square + err);
                f_x += _f * d_x / sqrt(d_square);
                f_y += _f * d_y / sqrt(d_square);
            }else{
                vx[i] = -vx[i];
                vy[i] = -vy[i];
            }
        }

        // collided with the boundary
        if((x[i] + sqrt(radius2) >= bound_x && vx[i] > 0) || (x[i] - sqrt(radius2) <= 0 && vx[i] < 0)){
            vx[i] = -vx[i];
        }
        if((y[i] + sqrt(radius2) >= bound_y && vy[i] > 0) || (y[i] - sqrt(radius2) <= 0 && vy[i] < 0)){
            vy[i] = -vy[i];
        }

        // update the velocity
        a_x = f_x / m[i];
        a_y = f_y / m[i];
        vx[i] += a_x * dt;
        vy[i] += a_y * dt;
    }
}


int main(int argc, char *argv[]) {
    n_body = atoi(argv[1]);
    n_iteration = atoi(argv[2]);

	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //pad the data
    int n_body_padded;
    if (n_body % world_size != 0)
        n_body_padded = n_body + world_size - (n_body % world_size); 
    else
        n_body_padded = n_body;
    
    double* m;
    double* x;
    double* y;
    double* vx;
    double* vy;

    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;

    Logger l = Logger("mpi", n_body, bound_x, bound_y);

    if (my_rank == 0) {
        m = new double[n_body_padded];
        x = new double[n_body_padded];
        y = new double[n_body_padded];
        vx = new double[n_body_padded];
        vy = new double[n_body_padded];
        generate_data(m, x, y, vx, vy, n_body, n_body_padded);
        
	}else{
            m = new double[n_body_padded];
            x = new double[n_body_padded];
            y = new double[n_body_padded];
    }

    int local_n = n_body_padded / world_size;
    double * local_m = new double[n_body_padded];
	double * local_x = new double[n_body_padded];
    double * local_y = new double[n_body_padded];
    double * local_vx = new double[n_body_padded];
    double * local_vy = new double[n_body_padded];

    // scatter the data
    MPI_Scatter(vx, local_n, MPI_DOUBLE, local_vx, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(vy, local_n, MPI_DOUBLE, local_vy, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(x, local_n, MPI_DOUBLE, local_x, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(y, local_n, MPI_DOUBLE, local_y, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(m, local_n, MPI_DOUBLE, local_m, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(m, n_body_padded, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // compute on local array
    for (int i=0; i<n_iteration; i++) {
        if (my_rank == 0) {
            t1 = std::chrono::high_resolution_clock::now();
        }
        MPI_Bcast(x, n_body_padded, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(y, n_body_padded, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        update_velocity(local_m, m, local_x, local_y, x, y, local_vx, local_vy, local_n);
        MPI_Barrier(MPI_COMM_WORLD);
        update_position(local_m, local_x, local_y, local_vx, local_vy, local_n);
        MPI_Barrier(MPI_COMM_WORLD);
        
        MPI_Gather(local_x, local_n, MPI_DOUBLE, x, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(local_y, local_n, MPI_DOUBLE, y, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (my_rank == 0) {
            t2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> time_span = t2 - t1;
            time_span_total += time_span;
            printf("Iteration %d, elapsed time: %.5f\n", i, time_span);
            printf("Total time: %.3f\n", i, time_span_total);

            l.save_frame(x, y);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

	if (my_rank == 0){
		printf("Student ID: 119010382\n"); // replace it with your student id
		printf("Name: Shiqi Yang\n"); // replace it with your name
		printf("Assignment 3: N Body Simulation MPI Implementation\n");
	}

	MPI_Finalize();

	return 0;
}

