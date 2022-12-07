#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#ifdef GUI
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "./headers/physics.h"
#include "./headers/logger.h"

int n_body;
int n_iteration;

std::chrono::duration<double> time_span_total;

void generate_data(double *m, double *x,double *y,double *vx,double *vy, int n) {
    // Generate proper initial position and mass for better visualization
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++) {
        m[i] = rand() % max_mass + 1.0f;
        x[i] = 2000.0f + rand() % (bound_x / 4);
        y[i] = 2000.0f + rand() % (bound_y / 4);
        vx[i] = 0.0f;
        vy[i] = 0.0f;
    }
}



void update_position(double *x, double *y, double *vx, double *vy, int n) {
    // update position 
    for(int i = 0; i < n; i++){
        x[i] += vx[i] * dt;
        y[i] += vy[i] * dt;
    }
}

void update_velocity(double *m, double *x, double *y, double *vx, double *vy, int n) {
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


void master() {
    double* m = new double[n_body];
    double* x = new double[n_body];
    double* y = new double[n_body];
    double* vx = new double[n_body];
    double* vy = new double[n_body];

    generate_data(m, x, y, vx, vy, n_body);

    Logger l = Logger("sequential", n_body, bound_x, bound_y);

    for (int i = 0; i < n_iteration; i++){
        
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        update_velocity(m, x, y, vx, vy, n_body);
        update_position(x, y, vx, vy, n_body);

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = t2 - t1;
        time_span_total += time_span;

        printf("Iteration %d, elapsed time: %.3f\n", i, time_span);
        printf("Total time: %.3f\n", i, time_span_total);

        l.save_frame(x, y);

        #ifdef GUI
        glClear(GL_COLOR_BUFFER_BIT);
        glColor3f(1.0f, 0.0f, 0.0f);
        glPointSize(2.0f);
        glBegin(GL_POINTS);
        double xi;
        double yi;
        for (int i = 0; i < n_body; i++){
            xi = x[i];
            yi = y[i];
            glVertex2f(xi, yi);
        }
        glEnd();
        glFlush();
        glutSwapBuffers();
        #else

        #endif
    }

    delete[] m;
    delete[] x;
    delete[] y;
    delete[] vx;
    delete[] vy;
    
}


int main(int argc, char *argv[]){
    
    n_body = atoi(argv[1]);
    n_iteration = atoi(argv[2]);

    #ifdef GUI
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(500, 500);
    glutCreateWindow("N Body Simulation Sequential Implementation");
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    gluOrtho2D(0, bound_x, 0, bound_y);
    #endif
    master();

    printf("Student ID: 119010382\n"); // replace it with your student id
    printf("Name: Shiqi Yang\n"); // replace it with your name
    printf("Assignment 2: N Body Simulation Sequential Implementation\n");
    
    return 0;
}


