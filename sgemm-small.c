#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <emmintrin.h>
#include <x86intrin.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <emmintrin.h>
#include <x86intrin.h>

void square_sgemm( int n, float *A, float *B, float *C ) {
  int f, g, h, i, j , k, l;
  int blockI = 64, blockJ = 64, blockK = 64;

  float temp, temp1, temp2, temp3, temp4;
  __m128 x;
  __m128 y;
  __m128 a;
  __m128 b;
  __m128 c;
  __m128 d;
  __m128 partialSum;
  __m128 partialSum1;
  __m128 partialSum2;
  __m128 partialSum3;
  __m128 partialSum4;
  __m128 partialSum5;
  __m128 partialSum6;
  __m128 partialSum7;
  float cij=0.0, cij1=0.0, cij2=0.0, cij3=0.0, cij4=0.0, cij5=0.0, cij6=0.0, cij7=0.0;
  __m128 c1;
  __m128 c2;
  float *At = malloc(n*n*sizeof(float));
  for (i = 0; i < n; i ++) {
      for (j = 0; j < n/4*4; j += 4) {

          x = _mm_loadu_ps(A + j + i*n);
          _MM_EXTRACT_FLOAT(temp, x, 0);
          At[i+j*n] = temp;
          _MM_EXTRACT_FLOAT(temp, x, 1);
          At[i+(j+1)*n] = temp;
          _MM_EXTRACT_FLOAT(temp, x, 2);
          At[i+(j+2)*n] = temp;
          _MM_EXTRACT_FLOAT(temp, x, 3);
          At[i+(j+3)*n] = temp;

      }
      for (; j<n; j++) {
          At[i + j*n] = A[j+i*n];
      }
  }

  for (g =0; g < n/blockI*blockI; g += blockI) {
      for (h = 0; h < n/blockJ*blockJ; h += blockJ) {
          for (f = 0; f < n/blockK*blockK; f += blockK) {
              // For each row i of A
              for (i = g; i < g+blockI; i+=4) {
                  // For each column j of B
                  for (j = h; j < h + blockJ; j+=2)
                    {
                      //load C inital values
                      c1 = _mm_loadu_ps(C + i + j*n);
                      c2 = _mm_loadu_ps(C + i + (j+1)*n);

                      //this will hold 4 floats which sum to the dot product
                      partialSum = _mm_setzero_ps();
                      partialSum1 = _mm_setzero_ps();
                      partialSum2 = _mm_setzero_ps();
                      partialSum3 = _mm_setzero_ps();
                      partialSum4 = _mm_setzero_ps();
                      partialSum5 = _mm_setzero_ps();
                      partialSum6 = _mm_setzero_ps();
                      partialSum7 = _mm_setzero_ps();

                      for(k = f; k < f + blockK, k < n; k += 4) {
                          a = _mm_loadu_ps(At + k + i*n);
                          x = _mm_loadu_ps(B + k + j*n);
                          y = _mm_mul_ps(a,x);
                          partialSum = _mm_add_ps(partialSum, y);
                          b = _mm_loadu_ps(At + k + (i+1)*n);
                          y = _mm_mul_ps(b,x);
                          partialSum1 = _mm_add_ps(partialSum1, y);
                          c = _mm_loadu_ps(At + k + (i+2)*n);
                          y = _mm_mul_ps(c,x);
                          partialSum2 = _mm_add_ps(partialSum2,y);
                          d = _mm_loadu_ps(At + k + (i+3)*n);
                          y = _mm_mul_ps(d,x);
                          partialSum3 = _mm_add_ps(partialSum3,y);
                          x = _mm_loadu_ps(B + k + (j+1)*n);
                          y = _mm_mul_ps(a, x);
                          partialSum4 = _mm_add_ps(partialSum4,y);
                          y = _mm_mul_ps(b, x);
                          partialSum5 = _mm_add_ps(partialSum5, y);
                          y = _mm_mul_ps(c, x);
                          partialSum6 = _mm_add_ps(partialSum6, y);
                          y = _mm_mul_ps(d, x);
                          partialSum7 = _mm_add_ps(partialSum7, y);

                      }

                      partialSum = _mm_hadd_ps(partialSum, partialSum1);
                      partialSum1 = _mm_hadd_ps(partialSum2, partialSum3);
                      partialSum2 = _mm_hadd_ps(partialSum, partialSum1); // [p0,p1,p2,p3] where p0 = sum of partialSum0

                      partialSum4 = _mm_hadd_ps(partialSum4, partialSum5);
                      partialSum5 = _mm_hadd_ps(partialSum6, partialSum7);
                      partialSum6 = _mm_hadd_ps(partialSum4, partialSum5); // [p4,p5,p6,p7]

                      c1 = _mm_add_ps(c1, partialSum2);
                      c2 = _mm_add_ps(c2, partialSum6);
                      _mm_storeu_ps(C + i + j*n, c1);
                      _mm_storeu_ps(C + i + (j+1)*n, c2);
                    }
              }
          }
      }
	}
	
	//everything cleanup
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			cij = C[i + j*n];
			for (k = f; k < n; k++) {
				cij += At[k + i*n] * B[k + j*n];
			}
			C[i + j*n] = cij;
		}
	}
	for (i = 0; i < g; i++) {
		for (j = h; j < n; j++) {
			cij = C[i + j*n];
			for (k = 0; k < f; k++) {
				cij += At[k + i*n] * B[k + j*n];
			}
			C[i + j*n] = cij;
		}
	}
	for (i = g; i < n; i++) {
		for (j = 0; j < n; j++) {
			cij = C[i + j*n];
			for (k = 0; k < f; k++) {
				cij += At[k + i*n] * B[k + j*n];
			}
			C[i + j*n] = cij;
		}
	}
  free(At);
}