#ifndef __PARVECTVECTMULT_H__
#define __PARVECTVECTMULT_H__

inline float
getVVal (const float *buf, int i,int vsize)                             // get value from vector
{
  size_t id = i;
  return buf[id];
}

inline void
setVVal (float *buf, int i, float val ,int vsize)                // set value in a vector
{
  size_t id = i ;
  buf[id] = val;
}



struct ParVectVectMult
{
  int vsize;
  float *vectorB;
  float *vectorA, *result_vector;

  void operator () (const blocked_range < size_t > &r) const           // operator function
  {
     int i;
         
     float ai,bi,sum;
        
        sum = 0.0;
    	for (i = r.begin (); i != r.end (); ++i)
      	{
            	ai = getVVal (vectorB,i,vsize);
                bi = getVVal (vectorA,i,vsize);
                sum += ai * bi;
		setVVal (result_vector,i, sum,vsize);
        }
     
  }
   
};

#endif

