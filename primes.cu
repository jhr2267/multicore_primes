#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define THREADS_PER_BLOCK 1024
#define TRILLION 1000000000000
#define FIRSTPRIMENUM 100  // first x primes to calculate manually

int firstprimes[FIRSTPRIMENUM];



//*************************************
//
//			Kernel
//
//*************************************

__global__ void trial_division_kernel(unsigned long long int n, int *ret){
	unsigned long long int i = 5 + 6 * (threadIdx.x + blockIdx.x * blockDim.x);
	if (i*i <= n){
		if (((n % i) == 0) || ((n % (i+2)) == 0))
			*ret = 1;
	}

	
}


//*************************************
//
//			Trial Division
//
//*************************************
// this is a slight optimization of most naieve algorithm
// see  https://en.wikipedia.org/wiki/Primality_test#Pseudocode
int trial_division(unsigned long long int n){
	if (n <= 1)
		return 0;
	else if (n <= 3)
		return 1;
	else if (((n % 2) == 0) || ((n % 3) == 0))
		return 0;

	long int root = sqrt(n);
	int ret= 0;
	int *d_ret;
	cudaMalloc((void **)&d_ret, sizeof(int));	
	
	trial_division_kernel<<<(root + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(n, d_ret);
	cudaMemcpy(&ret, d_ret, sizeof(int), cudaMemcpyDeviceToHost);

	if (ret)
		return 0;

	return 1;

}

//*************************************
//
//		Trial Division with Sieve
//
//*************************************

int trial_division_sieve(unsigned long long int n){
	if (n <= 1)
		return 0;
	else if (n <= 3)
		return 1;
	//else if (((n % 2) == 0) || ((n % 3) == 0))
	//	return 0;

	// the sieve makes the commented lines above redundant
	int i;
	for (int i = 0; i < P; i++){
		if ((n % firstprimes[i]) == 0)
			return 0;
	}

	long int root = sqrt(n);
	int ret= 0;
	int *d_ret;
	cudaMalloc((void **)&d_ret, sizeof(int));	
	
	trial_division_kernel<<<(root + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(n, d_ret);
	cudaMemcpy(&ret, d_ret, sizeof(int), cudaMemcpyDeviceToHost);	
	if (ret)
		return 0;

	return 1;

}

//*************************************
//
//			Helper Functions
//
//*************************************

int trial_division_c(unsigned long long int n){
	if (n <= 1)
		return 0;
	else if (n <= 3)
		return 1;
	else if (((n % 2) == 0) || ((n % 3) == 0))
		return 0;
	unsigned long long int i = 5;
	while (i*i <= n){
		if (((n % i) == 0) || ((n % (i+2)) == 0))
			return 0;
		i = i + 6;
	}
	return 1;

}

void init_firstprimes(){
	int i=2, numprime=0;
	while(numprime<FIRSTPRIMENUM){
		if (trial_division(i)){
			firstprimes[numprime] = 1;
			numprime ++;
			printf("%llu is prime \n", i);
		}
		i++;
	}
}



//*************************************
//
//			Main
//
//*************************************

int main(void){

	double time_spent = 0.0;
	clock_t begin = clock();

	init_firstprimes();
	
	unsigned long long int i;
	int numprimes = 0;
	for (i = TRILLION*1000; i < TRILLION*1000 + 10000; i++)
	//for (i = 0; i < 1000; i++)
		if (trial_division(i)){
			numprimes ++;
			//printf("%llu is prime \n", i);
		}



	clock_t end = clock();
	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Time elpased is %f seconds\n", time_spent);
	printf("In the 1000 trillion range average time %f\n", time_spent/10000);
	printf("number of primes in this range %d\n", numprimes);
}
