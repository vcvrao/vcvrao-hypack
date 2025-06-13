#ifndef __PARVECTVECTMULT_H__
#define __PARVECTVECTMULT_H__

inline float
getVVal (const float *buf, int i,int vsize)                 // get value from vector
{
  size_t id = i;
  return buf[id];
}

inline void
setVVal (float *buf, int i, float val ,int vsize)             // set value in vector
{
  size_t id = i ;
  buf[id] = val;
}



struct ParVectVectMult
{
  //size_t nrows;
  //size_t ncols;
  size_t vsize;
  float *vectorB;
  float *vectorA, *result_vector;

  void operator () (const blocked_range < size_t > &r) const         // operator function definition
  {
    int i;
    float ai,bi,sum;
    
    sum = 0.0;

    for (i = r.begin (); i != r.end (); ++i)
      {
		ai = getVVal (vectorB, i, vsize);
                bi = getVVal (vectorA, i ,vsize);
                sum += ai * bi;
           
           setVVal (result_vector,i, sum,vsize);                    // assign values to result vector
      }
  }
};

#endif

