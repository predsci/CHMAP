      subroutine ezseg(IMG,SEG,nt,np,thresh1,thresh2,nc,iters)
c
c-----------------------------------------------------------------------
c      ______ ______ _____ ______ _____ 
c     |  ____|___  // ____|  ____/ ____|
c     | |__     / /| (___ | |__ | |  __ 
c     |  __|   / /  \___ \|  __|| | |_ |
c     | |____ / /__ ____) | |___| |__| |
c     |______/_____|_____/|______\_____|
c     Multi-core CPU version using OpenMP
c
c     EZSEG: Routine to segment an image using a two-threshold
c     variable-connectivity region growing method.
c
c     INPUT/OUTPUT:
c        IMG:    Input image.
c        SEG: 
c             ON INPUT:
c                 Matrix of size (nt,np) which contain
c                 1's where there is valid IMG data, and
c                 non-zero values for areas with invalid/no IMG data.
c             ON OUTPUT:
c                 Segmentation map (0:detection, same as input o.w.).
c        nt,np:   Dimensions of image.
c        thresh1: Seeding threshold value.
c        thresh2: Growing threshold value.
c        nc:      # of consecutive pixels needed for connectivity.
c        iters:
c             ON INPUT: 
c                 maximum limit on number of iterations.
c             ON OUTPUT: 
c                 number of iterations performed.
c
c----------------------------------------------------------------------
c
c Copyright (c) 2015 Predictive Science Inc.
c 
c Permission is hereby granted, free of charge, to any person obtaining
c a copy of this software and associated documentation files 
c (the "Software"), to deal in the Software without restriction, 
c including without limitation the rights to use, copy, modify, merge,
c publish, distribute, sublicense, and/or sell copies of the Software,
c and to permit persons to whom the Software is furnished to do so, 
c subject to the following conditions:
c 
c The above copyright notice and this permission notice shall be 
c included in all copies or substantial portions of the Software.
c 
c THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
c EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
c MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
c NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
c BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
c ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
c CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
c SOFTWARE.
c
c----------------------------------------------------------------------
c
      implicit none
c
c----------------------------------------------------------------------
c
      real*4 :: thresh1,thresh2,tmp_sum
      real*4,dimension(nt,np) :: IMG
      real*4,dimension(nt,np) :: SEG_TMP,SEG
      real*4,dimension(15) :: local_vec
      integer :: max_iters,nt,np,val_modded
      integer :: i,j,k,ii,jj,iters,fillit,nc
c
c----------------------------------------------------------------------
c
c     Make copy of CH array:
      SEG_TMP=SEG
c
      max_iters=iters
      iters=0
c
      do k=1,max_iters
        val_modded=0
c$omp parallel do shared(SEG,SEG_TMP,thresh1,thresh2,val_modded)
c$omp& private(i,j,local_vec,fillit)
        do j=2,np-1
          do i=2,nt-1
c
            fillit=0
c
            !If in a no or bad data point, do nothing:
            if (SEG_TMP(i,j).eq.1) then
              !If value is within thresh1, mark ch pixel:
              if (IMG(i,j).le.thresh1) then
                fillit=1
                !If value is less than thresh2, check neighbors:
              else if (IMG(i,j).le.thresh2) then
                local_vec( 1)=SEG_TMP(i-1,j+1)
                local_vec( 2)=SEG_TMP(i  ,j+1)
                local_vec( 3)=SEG_TMP(i+1,j+1)
                local_vec( 4)=SEG_TMP(i+1,j  )
                local_vec( 5)=SEG_TMP(i+1,j-1)
                local_vec( 6)=SEG_TMP(i  ,j-1)
                local_vec( 7)=SEG_TMP(i-1,j-1)
                local_vec( 8)=SEG_TMP(i-1,j  )
                local_vec( 9)=local_vec(1)
                local_vec(10)=local_vec(2)
                local_vec(11)=local_vec(3)
                local_vec(12)=local_vec(4)
                local_vec(13)=local_vec(5)
                local_vec(14)=local_vec(6)
                local_vec(15)=local_vec(7)
c
                do ii=1,8
                  tmp_sum=0
                  do jj=1,nc
                    tmp_sum=tmp_sum+local_vec(ii+jj-1)
                  enddo
                  if (tmp_sum.eq.0) then
                    fillit=1
                    exit
                  endif
                enddo
              endif 
c
              if (fillit.eq.1) then
                SEG(i,j)=0
                if (val_modded.eq.0) then
                  val_modded=1
                endif
              endif
            endif
c
          enddo
        enddo
c$omp end parallel do
        iters=iters+1
        if (val_modded.eq.0) then
          exit
        endif
        !Reset tmp to be new map iterate:
        SEG_TMP=SEG
      enddo
c
      end subroutine
c
c----------------------------------------------------------------------
c 
