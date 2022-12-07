#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <pthread.h>

#ifdef GUI
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "./headers/physics.h"
#include "./headers/logger.h"

int n_thd; // number of threads

int n_body;
int n_iteration;

std::chrono::duration<double> time_span_total;

pthread_mutex_t mutex;


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



void update_position(double *x, double *y, double *vx, double *vy, int index_b, int index_e) {
    // update position 
    for(int i = index_b; i <= index_e; i++){
        x[i] += vx[i] * dt;
        y[i] += vy[i] * dt;
    }
}

void update_velocity(double *m, double *x, double *y, double *vx, double *vy, int n, int index_b, int index_e) {
    // calculate force and acceleration, update velocity
    double f_x;
    double f_y;
    double _f;
    double a_x;
    double a_y;
    double d_x;
    double d_y;
    double d_square;
    
    for(int i = index_b; i <= index_e; i++){
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


typedef struct {
    // specify your arguments for threads
    int p_id;
    double* m;
    double* x;
    double* y;
    double* vx;
    double* vy;
} Args;


void* worker(void* args) {
    // procedure in each threads
    
    Args* my_arg = (Args*) args;
    int p_id = my_arg->p_id;
    double* m = my_arg->m;
    double* x = my_arg->x;
    double* y = my_arg->y;
    double* vx = my_arg->vx;
    double* vy = my_arg->vy;

    int index_b, index_e;
    if(p_id != n_thd -1){
        index_b = p_id * (n_body / n_thd);
        index_e = index_b + (n_body / n_thd) - 1;
    }else{
        index_b = p_id * (n_body / n_thd);
        index_e = n_body - 1;
    }

    update_velocity(m, x, y, vx, vy, n_body, index_b, index_e);
    update_position(x, y, vx, vy, index_b, index_e);
}


void master(){
    double* m = new double[n_body];
    double* x = new double[n_body];
    double* y = new double[n_body];
    double* vx = new double[n_body];
    double* vy = new double[n_body];

    generate_data(m, x, y, vx, vy, n_body);

    Logger l = Logger("pthread", n_body, bound_x, bound_y);

    pthread_t thds[n_thd];
    Args args[n_thd];
    int local_size[n_thd];
    for(int i = 0; i < n_thd; i++){
        if(i != n_thd - 1)
            local_size[i] = n_body / n_thd;
        else
            local_size[i] = n_body / n_thd + n_body % n_thd;
    }

    for(int i = 0; i < n_thd; i++){
        args[i].p_id = i;
        args[i].m = m;
        args[i].x = x;
        args[i].y = y;
        args[i].vx = vx;
        args[i].vy = vy;
    }
    
    pthread_mutex_init(&mutex, NULL);


    for (int i = 0; i < n_iteration; i++){
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        for (int thd = 0; thd < n_thd; thd++) pthread_create(&thds[thd], NULL, worker, &args[thd]);
        for (int thd = 0; thd < n_thd; thd++) pthread_join(thds[thd], NULL);

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


int main(int argc, char *argv[]) {
    n_body = atoi(argv[1]);
    n_iteration = atoi(argv[2]);
    n_thd = atoi(argv[3]);

    #ifdef GUI
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(500, 500);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("Pthread");
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0, bound_x, 0, bound_y);
    #endif
    master();

	return 0;
}

