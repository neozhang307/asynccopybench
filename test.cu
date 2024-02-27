#include <cuda_runtime.h>
#include <stdio.h>

void __global__ simplestream(double* a, double*b, int N)
{
  int tid= threadIdx.x+blockIdx.x*blockDim.x;
  for(int id=tid; id<N; id+=blockDim.x*gridDim.x)
  {
    a[id]=b[id]+id;
  }
}

int  main(int argc, char const *argv[])
{
  cudaStream_t streamA, streamB;
  cudaStreamCreate ( &streamA);
  cudaStreamCreate ( &streamB);
  int N=1024*1024;
  int size=N*sizeof(double);
  double*ptr1;
  double*ptr2;
  double*ptr3;
  double*ptr4;
  double*h_ptr;
  cudaHostAlloc(&h_ptr, N*sizeof(double),cudaHostAllocDefault);
  dim3 threads = dim3(512, 1);
  dim3 blocks  = dim3(N / threads.x, 1);
  cudaEvent_t start1, stop1;
  cudaEvent_t start2, stop2;
  cudaEventCreate(&start1);
  cudaEventCreate(&stop1);
  cudaEventCreate(&start2);
  cudaEventCreate(&stop2);
  cudaEventRecord(start1, 0);
  cudaMallocAsync(&ptr1, size, streamA);
  cudaMallocAsync(&ptr2, size, streamA);
  cudaEventRecord(stop1, 0);
  simplestream<<<blocks, threads, 0, streamA>>>(ptr1,ptr2,N );
  cudaMemcpy(h_ptr, ptr2, size, cudaMemcpyHostToDevice);
  cudaFreeAsync(ptr1,streamA);
  cudaFreeAsync(ptr2,streamA);
  cudaEventRecord(start2, 0);
  cudaMallocAsync(&ptr3, size, streamB);
  cudaMallocAsync(&ptr4, size, streamB);
  cudaEventRecord(stop2, 0);
  simplestream<<<blocks, threads, 0, streamB>>>(ptr3,ptr4,N);
  cudaMemcpy(h_ptr, ptr4, size, cudaMemcpyHostToDevice);
  cudaFreeAsync(ptr3,streamB); 
  cudaFreeAsync(ptr4,streamB); 
  cudaFreeHost(h_ptr);
  float timer1,timer2;
  cudaEventElapsedTime ( &timer1,  start1,  stop1 );
  cudaEventElapsedTime ( &timer2,  start2,  stop2 );
  printf("%f,%f\n",timer1, timer2);
  cudaEventDestroy(start1);
  cudaEventDestroy(stop1);
  cudaEventDestroy(start2);
  cudaEventDestroy(stop2);
  cudaStreamDestroy(streamB);
  cudaStreamDestroy(streamA);
  return 0;

}

