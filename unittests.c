#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "tests/utest.h"

#include "linalg.h"
#include "mtxio.h"


double norm(double *y, double *x, int M)
{
    double diff = 0;
    for (int i = 0; i < M; i++)
    {
	diff += pow(x[i] - y[i], 2);
    }

    return sqrt(diff);
}


UTEST(mtxio, read_2d_mtx)
{

    double A_sol[] = {
	40, 35, 47,
	2, 42, 62,
	31, -28, 50,
	29, 11, -3
    };
    
    char filename[] = "tests/A_io.mtx";

    int size_0, size_1;
    double *A = read_2d_mtx(filename,
			     &size_0, &size_1);

    ASSERT_EQ(size_0, 4);
    ASSERT_EQ(size_1, 3);

    double error = norm(A, A_sol,size_0*size_1);
    
    ASSERT_LT(error, 10e-6);

    free(A);
    A = NULL;
}

UTEST(mtxio, read_1d_mtx)
{

    double b_sol[MAX_SIZE] = {2, 29, 12, 6};
    
    char filename[] = "tests/b_io.mtx";

    int size_0;
    double *b = read_1d_mtx(filename, &size_0);

    ASSERT_EQ(size_0, 4);
    
    double error = norm(b, b_sol, size_0);
    ASSERT_LT(error, 10e-6);

    free(b);
    b = NULL;
    
}

UTEST(mtxio, write_1d_mtx)
{
    char filename_in[] = "tests/b_io.mtx";
    int size_0;
    double *b = read_1d_mtx(filename_in, &size_0);

    char filename_out[] = "tests/b_tmp.mtx";
    write_1d_mtx(filename_out, b, size_0);

    char command[50];
    sprintf(command, "diff -w %s %s ",filename_in,
	    filename_out);
    int ret = system(command);
    ASSERT_TRUE(ret==0);
    
    sprintf(command, "rm %s",filename_out);
    
    ret = system(command);
    (void)ret;

    free(b);
    b = NULL;
    
}

UTEST(linalg, mult_ax)
{
    int ret = system("python tests/python_linalg_ax.py");
    char filename_in_A[] = "A_dax.mtx";
    char filename_in_x[] = "x_dax.mtx";
    char filename_in_sol[] = "sol_dax.mtx";

    int size_0, size_1;
    double *A = read_2d_mtx(filename_in_A, &size_0, &size_1);

    int tmp;
    double *x = read_1d_mtx(filename_in_x, &tmp);
    double *sol = read_1d_mtx(filename_in_sol, &tmp);
    
    double *out = (double*)calloc((size_t)(size_0),
				  sizeof(double));
    mult_ax(out, A, x, size_0, size_1);

    double error = norm(sol, out, size_0);

    ASSERT_LT(error, 10e-6);

    ret = system("rm A_dax.mtx x_dax.mtx sol_dax.mtx");
    (void)ret;

    free(A);
    free(x);
    free(sol);
    free(out);
    
    A = NULL;
    x = NULL;
    sol = NULL;
    out = NULL;
}

UTEST(linalg, mult_ata)
{
    int ret = system("python tests/python_linalg_ata.py");
    char filename_in_A[] = "A_ata.mtx";
    char filename_in_sol[] = "sol_ata.mtx";

    int size_0, size_1;
    double *A = read_2d_mtx(filename_in_A, &size_0, &size_1);

    int size_0_sol, size_1_sol;
    double *sol = read_2d_mtx(filename_in_sol,
			      &size_0_sol, &size_1_sol);

    double *out = (double*)calloc((size_t)(size_1*size_1),
				  sizeof(double));
    mult_ata(out, A, &size_0, &size_1);

    ASSERT_EQ(size_0, size_0_sol);
    ASSERT_EQ(size_1, size_1_sol);

    double error = norm(out, sol, size_0*size_1);
    
    ASSERT_LT(error, 10e-6);
    
    char command[50];
    sprintf(command,"rm %s %s",
		    filename_in_A, filename_in_sol);
    ret = system(command);
    (void)ret;

    free(A);
    free(sol);
    free(out);
    A = NULL;
    sol = NULL;
    out = NULL;
}


UTEST(linalg, mult_atx)
{
    int ret = system("python tests/python_linalg_atx.py");
    char filename_in_A[] = "A_atx.mtx";
    char filename_in_x[] = "x_atx.mtx";
    char filename_in_sol[] = "sol_atx.mtx";

    int size_0, size_1;
    double *A = read_2d_mtx(filename_in_A, &size_0, &size_1);

    int temp;
    double *x = read_1d_mtx(filename_in_x, &temp);

    int size_0_sol;
    double *sol = read_1d_mtx(filename_in_sol, &size_0_sol);

    double *out = (double*)calloc((size_t)size_1, sizeof(double));
    mult_atx(out, A, x, &size_0, &size_1);
    
    ASSERT_EQ(size_0, size_0_sol);
    
    double error = norm(out, sol, size_0);

    ASSERT_LT(error, 10e-6);
    
    char command[50];
    sprintf(command,"rm %s %s %s",
	    filename_in_A, filename_in_x,
	    filename_in_sol);
    ret = system(command);
    (void)ret;

    free(A);
    free(x);
    free(sol);
    free(out);
    A = NULL;
    x = NULL;
    sol = NULL;
    out = NULL;
    
}


UTEST(linalg, mult_xy)
{
    int ret = system("python tests/python_linalg_mxy.py > tmp.txt");
    char filename_in_x[] = "x_dax.mtx";
    char filename_in_y[] = "y_dax.mtx";

    int size_0;
    double *x = read_1d_mtx(filename_in_x, &size_0);

    int tmp;
    double *y = read_1d_mtx(filename_in_y, &tmp);

    double out = mult_xy(y, x, size_0);

    double sol;
    FILE *fp = fopen("tmp.txt","r");
    if (fscanf(fp,"%lf",&sol) == -1)
    {
	printf("ERROR: fscanf() failed\n");
	exit(1);
    }

    fclose(fp);
    
    double diff = fabs((out - sol)/sol);

    ASSERT_LT(diff,
	      10e-12);

    char command[50];
    sprintf(command,"rm tmp.txt %s %s",
	    filename_in_x, filename_in_y);
    ret = system(command);
    (void)ret;

    free(x);
    free(y);
    x = NULL;
    y = NULL;	
}

UTEST(linalg, add_xy)
{
    int ret = system("python tests/python_linalg_axy.py");
    char filename_in_x[] = "x_axy.mtx";
    char filename_in_y[] = "y_axy.mtx";
    char filename_in_c[] = "c_axy.mtx";
    char filename_in_sol[] = "sol_axy.mtx";

    int size_0;
    double *x = read_1d_mtx(filename_in_x, &size_0);

    int tmp;
    double *y = read_1d_mtx(filename_in_y, &tmp);
    double *c = read_1d_mtx(filename_in_c, &tmp);
    double *sol = read_1d_mtx(filename_in_sol, &tmp);
 
    double *out = (double*)calloc((size_t)size_0,
				  sizeof(double));

    add_xy(out, x, y, c[0], size_0);

    double error = norm(out, sol, size_0);

    ASSERT_LT(error, 10e-6);
    
    char command[50];
    sprintf(command,"rm %s %s %s %s",
	    filename_in_x, filename_in_y,
	    filename_in_c, filename_in_sol);
    ret = system(command);
    (void)ret;

    free(x);
    free(y);
    free(c);
    free(sol);
    free(out);

    x = NULL;
    y = NULL;
    c = NULL;
    sol = NULL;
    out = NULL;
}

UTEST(linalg, solver)
{
    int ret = system("python tests/python_linalg_solve.py");
    char filename_in_A[] = "A_solve.mtx";
    char filename_in_b[] = "b_solve.mtx";
    char filename_in_sol[] = "sol_solve.mtx";

    int size_0, size_1;
    double *A = read_2d_mtx(filename_in_A, &size_0, &size_1);

    int tmp;
    double *b = read_1d_mtx(filename_in_b, &tmp);

    int size_sol;
    double *sol = read_1d_mtx(filename_in_sol, &size_sol);

    double *x = (double*)calloc((size_t)size_1, sizeof(double));
    cg_solver(x, A, b, size_0, size_1);

    double error = norm(x, sol, size_sol);

    ASSERT_LT(error, 10e-6);
    
    char command[50];
    sprintf(command,"rm %s %s %s",
	    filename_in_A, filename_in_b,
	    filename_in_sol);
    ret = system(command);
    (void)ret;

    free(A);
    free(b);
    free(sol);
    free(x);

    A = NULL;
    b = NULL;
    sol = NULL;
    x = NULL;

}

UTEST(main, usage)
{
    int out = system("./cg_solver > tmp_1.txt");

    ASSERT_EQ(out, 0);

    FILE *fp = fopen("tmp_2.txt","w");

    fprintf(fp,"Usage:\n    $ ./cg_solver <A_file> <b_file> "
	    "<x_file>\n");
    fclose(fp);

    out = system("diff -w tmp_1.txt tmp_2.txt");

    ASSERT_EQ(out, 0);

    int ret = system("rm tmp_1.txt tmp_2.txt");
    (void)ret;
    
}

UTEST(main, output)
{

    int ret = system("python tests/python_linalg.py");
    int out = system("./cg_solver A_test.mtx b_test.mtx "
		     "x_sol.mtx > tmp_2.txt");

    ASSERT_EQ(out, 0);

    char mat_size_str[10];
    
    FILE *fp = fopen("tmp_2.txt","r");
    if (fscanf(fp,"%*s %s %*s",mat_size_str) == -1)
    {
	printf("ERROR:  fscanf() failed to read file.\n");
	exit(1);
    }

    int n = 0;
    while (*(mat_size_str+n) != 'x')
    {
	n++;
    }

    char size_0_str[5];
    char size_1_str[5];

    strncpy(size_0_str, mat_size_str, (size_t)n);
    strcpy(size_1_str, mat_size_str+n+1);
    
    int size_0 = atoi(size_0_str);
    int size_1 = atoi(size_1_str);

    double time;
    if (fscanf(fp,"%*s %lf %*s",&time) == -1)
    {
    	printf("ERROR:  fscanf() failed to read file.\n");
    	exit(1);
	    
    }
    fclose(fp);
    
    ASSERT_LT(time, 2);

    FILE *fout = fopen("tmp_3.txt","w");
    fprintf(fout,"solving %ix%i system\n",
    	    size_0,size_1);
    fprintf(fout,"done...took %f seconds",time);
    fclose(fout);

    out = system("diff -w tmp_2.txt tmp_3.txt");
    ASSERT_EQ(out, 0);

    ret = system("rm A_test.mtx b_test.mtx x_test.mtx "
		 "tmp_2.txt tmp_3.txt x_sol.mtx");
    (void)ret;

}


UTEST_MAIN();
