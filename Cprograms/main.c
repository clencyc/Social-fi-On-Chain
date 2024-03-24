#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <omp.h>

#define MAX_POINTS 2000


int read_file(float points[], char* filename) {
    int n = 0;
    float next[2];
    FILE* file_ptr;
    file_ptr = fopen(filename,"r");
    if (file_ptr == NULL) {
        printf ("error : could not open file %s for reading\n",filename);
        exit(1);
    }
    while (fscanf (file_ptr,"%f %f",next,next+1) == 2) {
        if (n < MAX_POINTS) {
            points[2*n] = next[0];
            points[2*n+1] = next[1];
            n += 1;
        } else {
            printf ("Too many points in file %s\n",filename);
            fclose (file_ptr);
            exit(1);
        }
    }
    fclose (file_ptr);
    return n;
}

// calculate the distance squared between two points
float calc_dist_sq (float u[], float v[]) {
    float diff_x = u[0] - v[0];
    float diff_y = u[1] - v[1];
    return (diff_x*diff_x + diff_y*diff_y);
}

typedef struct centers_s {
    int c[5];
} centers_type;

// compute the cost squared for centers in the centers array
float center_cost_sq (float dist_sqs[], int n, centers_type check, float min_cost_sq) {
    float cost_sq = 0;
    for (int i=0;i<n;i++) {
        float dist_sq_1 = dist_sqs[i*n+check.c[0]];
        float dist_sq_2 = dist_sqs[i*n+check.c[1]];
        float dist_sq_3 = dist_sqs[i*n+check.c[2]];
        float dist_sq_4 = dist_sqs[i*n+check.c[3]];
        float dist_sq_5 = dist_sqs[i*n+check.c[4]];
        float min_dist_sq = dist_sq_1;
        if (dist_sq_2 < min_dist_sq) {
            min_dist_sq = dist_sq_2;
        }
        if (dist_sq_3 < min_dist_sq) {
            min_dist_sq = dist_sq_3;
        }
        if (dist_sq_4 < min_dist_sq) {
            min_dist_sq = dist_sq_4;
        }
        if (dist_sq_5 < min_dist_sq) {
            min_dist_sq = dist_sq_5;
        }
        // check to see if we can abort early
        if (min_dist_sq > min_cost_sq) {
            return min_dist_sq;
        }
        if (min_dist_sq > cost_sq) {
            cost_sq = min_dist_sq;
        }
    }
    return cost_sq;
}

double solve_5center(float dist_sqs[], int n, centers_type* optimal, long int* num_checked) {
    double min_cost_sq = DBL_MAX;
    long int total_num_checked = 0; // Use a local variable for reduction
    // Create an array to store the number of checks performed by each thread
    long int *num_checked_array = calloc(omp_get_max_threads(), sizeof(long int));

    #pragma omp parallel reduction(min:min_cost_sq) reduction(+:total_num_checked)
    {
        int thread_num = omp_get_thread_num();
        double local_min_cost_sq = DBL_MAX;
        centers_type local_optimal;
        long int local_num_checked = 0; // Initialize a local counter for each thread

        #ifdef DYNAMIC
        #pragma omp for schedule(dynamic) nowait
        #else
        #pragma omp for schedule(static) nowait
        #endif
        for (int i = 0; i < n - 4; i++) {
            for (int j = i + 1; j < n - 3; j++) {
                for (int k = j + 1; k < n - 2; k++) {
                    for (int l = k + 1; l < n - 1; l++) {
                        for (int m = l + 1; m < n; m++) {
                            local_num_checked++; // Increment the local counter
                            centers_type check = {i, j, k, l, m};
                            double cost_sq = center_cost_sq(dist_sqs, n, check, local_min_cost_sq);
                            if (cost_sq < local_min_cost_sq) {
                                local_min_cost_sq = cost_sq;
                                local_optimal = check;
                            }
                        }
                    }
                }
            }
        }

        #pragma omp critical
        {
            total_num_checked += local_num_checked; // Accumulate the local counts
            if (local_min_cost_sq < min_cost_sq) {
                min_cost_sq = local_min_cost_sq;
                *optimal = local_optimal;
            }
        }
        // Store the local count in the array at the index corresponding to the thread number
        num_checked_array[thread_num] = local_num_checked;
    }
       // Print the results in ascending order of thread numbers
    for (int i = 1; i < omp_get_num_threads(); i++) {
        printf("thread_num %d checked %ld 5-tuples\n", i, num_checked_array[i]);
    }
    free(num_checked_array); // frees the allocated memory

    *num_checked = total_num_checked; // Update the output variable
    return sqrt(min_cost_sq); // Return the minimal cost
}

int main (int argc, char** argv) {

    // the points array
    float points[2*MAX_POINTS];

    // read filename and thread count from command line
    if (argc < 3) {
        printf ("Command usage : %s %s %s\n",argv[0],"filename","thread_count");
        return 1;
    }
    int thread_count = atoi(argv[2]);
    omp_set_num_threads(thread_count);

    // read dataset
    int n = read_file (points,argv[1]);

    // calculate the distances squared table
    float dist_sqs[n*n];
    for (int i=0;i<n;i++) {
        for (int j=0;j<n;j++) {
            dist_sqs[i*n+j] = calc_dist_sq (points+2*i,points+2*j);
        }
    }

    // print the number of points and number of threads
    printf ("number of points = %d, number of threads = %d\n",n,thread_count);

    // start the timer
    double start_time, end_time;
    start_time = omp_get_wtime();

    // solve the 5-center problem exactly
    centers_type optimal;
    long int num_checked;
    double min_cost = solve_5center (dist_sqs,n,&optimal,&num_checked);

    // stop the timer
    end_time = omp_get_wtime();
    double elapsed = end_time-start_time;

    // print out the number of 5-tuples checked
    printf ("Total 5-tuples checked = %ld\n",num_checked);

    // print the minimal cost for the 5-center problem
    printf ("minimal cost = %g\n",min_cost);

    // print an optimal solution to the 5-center problem
    printf ("optimal centers : %d %d %d %d %d\n",optimal.c[0],optimal.c[1],
            optimal.c[2],optimal.c[3],optimal.c[4]);

    // print the elapsed time
    printf ("elapsed time = %g seconds\n",elapsed);

    // print the tuples checked per second
    printf ("5-tuples checked per second = %.0lf\n",num_checked/elapsed);

}