#include "cuda.h"
/*---------------------------------------------------------------------
   ______ ______ _____ ______ _____ 
  |  ____|___  // ____|  ____/ ____|
  | |__     / /| (___ | |__ | |  __ 
  |  __|   / /  \___ \|  __|| | |_ |
  | |____ / /__ ____) | |___| |__| |
  |______/_____|_____/|______\_____|
  GPU-enabled version using CUDA
  Version 1.01

  EZSEG: Routine to segment an image using a two-threshold
  variable-connectivity region growing method utilizing
  GPU acceleration through CUDA.

  void ezseg_cuda(float *IMG, float *SEG, int nt, int np,
                  float thresh1, float thresh2, int nc, int* iters)
 
  INPUT/OUTPUT:
        IMG:    Input image.
        SEG: 
             ON INPUT:
                 Matrix of size (nt,np) which contain
                 1's where there is valid IMG data, and
                 non-zero values for areas with invalid/no IMG data.
             ON OUTPUT:
                 Segmentation map (0:detection ,same as input o.w.).
        nt,np:   Dimensions of image.
        thresh1: Seeding threshold value.
        thresh2: Growing threshold value.
        nc:      # of consecutive pixels needed for connectivity.
        iters:
             ON INPUT: 
                 maximum limit on number of iterations.
             ON OUTPUT: 
                 number of iterations performed.

----------------------------------------------------------------------
 Copyright (c) 2015 Predictive Science Inc.
 
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files 
 (the "Software"), to deal in the Software without restriction, 
 including without limitation the rights to use, copy, modify, merge,
 publish, distribute, sublicense, and/or sell copies of the Software,
 and to permit persons to whom the Software is furnished to do so, 
 subject to the following conditions:
 
 The above copyright notice and this permission notice shall be 
 included in all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
 CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 ----------------------------------------------------------------------
*/

/*Define block size.*/
const int BLOCK_SIZE = 16;

/*** Kernel for ezseg iteration ***/

__global__ void ezseg_kernel(float *EUV, float *CHM, float *CHM_TMP, int nt, int np,
                             float thresh1, float thresh2, int nc, int *val_modded)
{
  int fillit,ij,ii,jj;

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  
  float local_vec[15],tmp_sum;

  if((i>0 && i<nt-1) && (j>0 && j<np-1))
  {
    ij = nt*j+i;  /*Location of i,j in 1D global arrays*/ 

    fillit = 0;   

    if(CHM_TMP[ij] == 1 ){  /*Good data not marked yet*/
      
      if(EUV[ij] <= thresh1){
        fillit = 1;
      }
      else if(EUV[ij] <= thresh2){                             
        local_vec[ 0] = CHM_TMP[nt*(j+1)+(i-1)];
        local_vec[ 1] = CHM_TMP[nt*(j+1)+(i  )];
        local_vec[ 2] = CHM_TMP[nt*(j+1)+(i+1)];
        local_vec[ 3] = CHM_TMP[nt*(j  )+(i+1)];
        local_vec[ 4] = CHM_TMP[nt*(j-1)+(i+1)];
        local_vec[ 5] = CHM_TMP[nt*(j-1)+(i  )];
        local_vec[ 6] = CHM_TMP[nt*(j-1)+(i-1)];
        local_vec[ 7] = CHM_TMP[nt*(j  )+(i-1)];
        local_vec[ 8] = local_vec[0];
        local_vec[ 9] = local_vec[1];
        local_vec[10] = local_vec[2];
        local_vec[11] = local_vec[3];
        local_vec[12] = local_vec[4];
        local_vec[13] = local_vec[5];
        local_vec[14] = local_vec[6];

        for(ii=0;ii<8;ii++){  
          tmp_sum = 0.0f;
          for(jj=0;jj<nc;jj++){
            tmp_sum = tmp_sum + local_vec[ii+jj];
          }
          if(tmp_sum == 0){
            fillit = 1;
            break;
          }
        }
      }  /*euv<thresh2*/

      if (fillit == 1) {
        CHM[ij] = 0.0f;
        if(*val_modded == 0) {
          atomicAdd(val_modded, 1);
        }
      } 
    } /*good data no mark*/
  } /*valid point*/
}

/*********************************************************************/
/*********************************************************************/
/*********************************************************************/

extern "C" void ezseg_cuda(float *EUV, float *CHM, int nt, int np,
                           float thresh1, float thresh2, int nc, int* iters )
{
    int val_modded,max_iters,k;

    /*GPU variables:*/
    float *EUVgpu,*CHMgpu,*CHM_TMPgpu;
    int *val_modded_gpu;

    /*Allocate GPU arrays:*/
    cudaMalloc((void **) &EUVgpu,     sizeof(float)*nt*np);
    cudaMalloc((void **) &CHMgpu,     sizeof(float)*nt*np);
    cudaMalloc((void **) &CHM_TMPgpu, sizeof(float)*nt*np);
    cudaMalloc((void **) &val_modded_gpu, sizeof(int));

    /*Copy euv and chm to GPU*/
    cudaMemcpy(EUVgpu,     EUV, sizeof(float)*nt*np, cudaMemcpyHostToDevice);
    cudaMemcpy(CHMgpu,     CHM, sizeof(float)*nt*np, cudaMemcpyHostToDevice);
    cudaMemcpy(CHM_TMPgpu, CHMgpu, sizeof(float)*nt*np, cudaMemcpyDeviceToDevice);

    /*Set up CUDA grid*/
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid((int)ceil((nt+0.0)/dimBlock.x),
                 (int)ceil((np+0.0)/dimBlock.y));

    /*Start main loop*/ 
    max_iters = *iters;
    *iters = 0;
    for(k=0;k<max_iters;k++){  
      /*Reset val_modded*/
      val_modded = 0;
      cudaMemcpy(val_modded_gpu, &val_modded, sizeof(int), cudaMemcpyHostToDevice);

      /*Perform iteration:*/
      ezseg_kernel<<<dimGrid,dimBlock>>>(EUVgpu,CHMgpu,CHM_TMPgpu,nt,np,thresh1,
                                           thresh2,nc,val_modded_gpu);

      *iters = *iters + 1;

      /*Make sure everything is done*/
      cudaDeviceSynchronize();

      /*Get data mod flag*/        
      cudaMemcpy(&val_modded, val_modded_gpu, sizeof(int), cudaMemcpyDeviceToHost);

      /*If no new CH points, break out of iterations*/
      if(val_modded == 0){
        break;
      }

      /*Reset tmp to be new map iterate for next iteration:*/
      cudaMemcpy(CHM_TMPgpu,CHMgpu,sizeof(float)*nt*np,cudaMemcpyDeviceToDevice);

    }

    /*Copy result from GPU back to CPU*/
    cudaMemcpy(CHM,CHMgpu,sizeof(float)*nt*np,cudaMemcpyDeviceToHost);

    /*Free up GPU memory*/
    cudaFree(EUVgpu);
    cudaFree(CHMgpu);
    cudaFree(CHM_TMPgpu);
    cudaFree(val_modded_gpu);
	
}
