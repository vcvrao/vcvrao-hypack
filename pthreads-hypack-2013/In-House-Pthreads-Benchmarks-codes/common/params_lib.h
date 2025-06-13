/* Functions required by write_params.c */
#include <sys/types.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <time.h>


#define FILENAME "input_paramaters.h"
#define LINE_LENGTH 400
#define MAKE_IN "../config/Make.inc"
#define DEFAULT_VALUE "(none)"
#define MAX_LENGTH 46
 
 FILE *makeinc;
 int get_args(int argc, char *argv[], int *nthreads, char *class, char *benchmark);
 int check_class(char class);
 int write_compiler_lib_info(FILE *fp);
 void check_line(char *line, char *label, char *val);
 void put_string(FILE *fp, char *name, char *val);
 int write_header(FILE *fp,char *benchmark,char class,int nthreads);

 /* Read the Command line arguments and verify the arguments */
 int get_args(int argc,char *argv[], int *nthreads,char *class,char *benchmark)
     {
      if (argc < 4)
         {
          printf("Usage: %s benchmark-name nthreads class\n", argv[0]);
          exit(1);
         }
      /* Getting the numnber of threads */
      *nthreads = atoi(argv[2]);
      /* Check the number of threads */
       if(atoi(argv[2]) <= 0 || atoi(argv[2]) > 8 )
         {
          printf("\n  Error : Number of threads must be 1\\2\\4\\8.\n ");
          exit(-1);
         }
       if(atoi(argv[2]) == 3 || atoi(argv[2]) == 5 || atoi(argv[2]) == 6 || atoi(argv[2]) == 7)
         {
          printf("\n  Error : Number of threads must be 1\\2\\4\\8.\n ");
          exit(-1);
         }

      /* Getting the class value i.e A/B/C */
      *class = *argv[3];

      /* Getting the benchmark and verifying */
       if(!strcmp(argv[1], "pi") || !strcmp(argv[1], "PI")) strcpy(benchmark,"pi");
       else if  (!strcmp(argv[1], "mat_cmp_db") || !strcmp(argv[1], "MAT_CMP_DB")) strcpy(benchmark,"mat_cmp_db");
       else if  (!strcmp(argv[1], "mat_cmp_in") || !strcmp(argv[1], "MAT_CMP_IN")) strcpy(benchmark,"mat_cmp_in");
       else if  (!strcmp(argv[1], "jacobi") || !strcmp(argv[1], "JACOBI")) strcpy(benchmark,"jacobi");
       else if  (!strcmp(argv[1], "int_sort") || !strcmp(argv[1], "INT_SORT")) strcpy(benchmark,"int_sort");
       else
          {
          printf("\n  Error : Invalid benchmark %s.",argv[1]);
          printf("\n Available benchmarks \n\t pi \n\t mat_cmp_db \n\t mat_cmp_in \n\t jacobi\n\t int_sort ");
          exit(-1);
          }
      return 0 ;
     }

 int write_header(FILE *fp,char *benchmark,char class,int nthreads)
     {
      fprintf(fp,"#define BENCHMARK %s",benchmark);
      fprintf(fp,"\n#define CLASS '%c'",class);
      fprintf(fp,"\n#define THREADS %d \n",nthreads);
      write_compiler_lib_info(fp);
      fclose(fp);
      return 0;
     }
 int check_class(char class)
   {
    /* Verifying tihe CLASS for either A/B/C */
    if(class != 'A' &&  class != 'B' && class != 'C')
      {
       printf("\n Error : Invalid benchmark class %c\n", class);
       printf("  Allowed classes are \"A\", \"B\" and  \"C\" \n");
       exit(1);
      }
     return 0;
   }


#define MAX_LENGTH 46
int write_compiler_lib_info(FILE *fp)
{
  char line[LINE_LENGTH];
  char compiletime[LINE_LENGTH];
  char cc[LINE_LENGTH], cflags[LINE_LENGTH], clink[LINE_LENGTH], clinkflags[LINE_LENGTH],
       c_lib[LINE_LENGTH], c_inc[LINE_LENGTH];
  struct tm *tmp;
  time_t t;
  makeinc = fopen(MAKE_IN, "r");
  if (makeinc == NULL) {
    printf("\n Error: File %s doesn't exist. %s is required to build the benchmarks.\
           Use the file config/make.def.template to build the %s\n", MAKE_IN,MAKE_IN,MAKE_IN);
    exit(1);
  }
  strcpy(cc, DEFAULT_VALUE);
  strcpy(cflags, DEFAULT_VALUE);
  strcpy(clink, DEFAULT_VALUE);
  strcpy(clinkflags, DEFAULT_VALUE);
  strcpy(c_lib, DEFAULT_VALUE);
  strcpy(c_inc, DEFAULT_VALUE);

  while (fgets(line, LINE_LENGTH, makeinc) != NULL)
  {
    if (*line == '#') continue;
    check_line(line, "CC", cc);
    check_line(line, "CFLAGS", cflags);
    check_line(line, "CLINK", clink);
    check_line(line, "CLINKFLAGS", clinkflags);
    check_line(line, "C_LIB", c_lib);
    check_line(line, "C_INC", c_inc);
  }


  (void) time(&t);
  tmp = localtime(&t);
  (void) strftime(compiletime, (size_t)LINE_LENGTH, "%d %a %b %Y %T", tmp);

          put_string(fp, "COMPILETIME", compiletime);
          put_string(fp, "CC", cc);
          put_string(fp, "CFLAGS", cflags);
          put_string(fp, "CLINK", clink);
          put_string(fp, "CLINKFLAGS", clinkflags);
          put_string(fp, "C_LIB", c_lib);
          put_string(fp, "C_INC", c_inc);
  return 0;
 }

void check_line(char *line, char *label, char *val)
{
  char *original_line;
  int n;
  original_line = line;
  while (*label != '\0' && *line == *label) {
    line++; label++;
  }
  if (*label != '\0') return;
  if (!isspace(*line) && *line != '=') return ;
  while (isspace(*line)) line++;
  if (*line != '=') return;
  while (isspace(*++line));
  if (*line == '\0') return;
  strcpy(val, line);
  n = strlen(val)-1;
  val[n--] = '\0';
  while (val[n] == '\\' && fgets(original_line, LINE_LENGTH, makeinc)) {
     line = original_line;
     while (isspace(*line)) line++;
     if (isspace(*original_line)) val[n++] = ' ';
     while (*line && *line != '\n' && n < LINE_LENGTH-1)
       val[n++] = *line++;
     val[n] = '\0';
     n--;
  }
}
int check_include_line(char *line, char *filename)
{
  char *include_string = "include";
  while (*include_string != '\0' && *line == *include_string) {
    line++; include_string++;
  }
  if (*include_string != '\0') return(0);
  if (!isspace(*line)) return(0);
  while (isspace(*++line));
  if (*line == '\0') return(0);
  while (*filename != '\0' && *line == *filename) {
    line++; filename++;
  }
  if (*filename != '\0' ||
      (*line != ' ' && *line != '\0' && *line !='\n')) return(0);
  else return(1);
}


#define MAX_LENGTH 46

void put_string(FILE *fp, char *name, char *val)
{
  int len;
  len = strlen(val);
  if (len > MAX_LENGTH) {
val[MAX_LENGTH] = '\0';
    val[MAX_LENGTH-1] = '.';
    val[MAX_LENGTH-2] = '.';
    val[MAX_LENGTH-3] = '.';
 len = MAX_LENGTH;
  }
  fprintf(fp, "#define %s \"%s\"\n", name, val);
}




