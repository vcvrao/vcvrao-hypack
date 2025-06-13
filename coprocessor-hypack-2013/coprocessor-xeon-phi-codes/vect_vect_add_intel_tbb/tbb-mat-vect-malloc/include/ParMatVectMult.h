#ifndef __PARMATVECTMULT_H__
#define __PARMATVECTMULT_H__


inline void
setVVal (float *buf, int i, float val ,int nrows)             // set value in vector
{
  size_t id = i ;
  buf[id] = val;
}

inline float
getVVal (const float *buf, int i,int nrows)                 // get value from vector
{
  size_t id = i;
  return buf[id];
}

inline float
getMVal (const float *buf, int i, int j ,int nrows ,int ncols)
{
  size_t id = j * nrows + i;                                           // get value from matrix
  return buf[id];
}




struct ParMatVectMult
{
  size_t nrows;
  size_t ncols;
  size_t vsize;
  float *matrixA;
  float *vectorA, *result_vector;

  void operator () (const blocked_range < size_t > &r) const         // operator function definition
  {
    int i, j, k;
    float aij,bj,sum;

    for (i = r.begin (); i != r.end (); ++i)
      {
	  sum = 0.0;            
          for (j = 0; j < ncols; ++j)
          {
		aij = getMVal (matrixA, i, j,nrows,ncols);
                bj = getVVal (vectorA,j,nrows);
                sum += aij * bj;
          } 
           
           setVVal (result_vector,i, sum,nrows);                    // assign values to result vector
      }
  }
};

#endif

