#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define THREADS_PER_BLOCK 1024
#define TRILLION 1000000000000


__global__ void trial_division_kernel(unsigned long long int n, int *ret){
	unsigned long long int i = 5 + 6 * (threadIdx.x + blockIdx.x * blockDim.x);
	__shared__ int local_ret;
	if (threadIdx.x == 0){
		local_ret = 0;
	}
	if (i*i <= n){
		if (((n % i) == 0) || ((n % (i+2)) == 0))
			atomicAdd(&local_ret, 1);
	}
	__syncthreads();
	
	if (threadIdx.x == 0)
		atomicAdd(ret, local_ret);
	
}





// method one
// this is a slight optimization of most naieve algorithm
// see  https://en.wikipedia.org/wiki/Primality_test#Pseudocode
int trial_division(unsigned long long int n){
	//printf("number is %d\n", n);
	if (n <= 1)
		return 0;
	else if (n <= 3)
		return 1;
	else if (((n % 2) == 0) || ((n % 3) == 0))
		return 0;
	//unsigned long long int i = 5;


	// make this parallel 
	// while (i*i <= n){
	// 	if (((n % i) == 0) || ((n % (i+2)) == 0))
	// 		return 0;
	// 	i = i + 6;
	// }

	long int root = sqrt(n);
	int ret= 0;
	//long long int *d_n;
	int *d_ret;
	//cudaMalloc((void **)&d_n, sizeof(long long int));
	//cudaMemcpy(d_n, n, sizeof(long long int), cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_ret, sizeof(int));	

	trial_division_kernel<<<(root + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(n, d_ret);
	cudaMemcpy(&ret, d_ret, sizeof(int), cudaMemcpyDeviceToHost);
	//cudaFree(d_n); 
	//printf("returned %d\n",ret);	
	if (ret)
		return 0;

	return 1;

}

int wilsons_algorithm(unsigned long long int n){
return 0;
}


// idea- prepare a large table via sieve of eratosthenes in parallel
// use that with trial division method



int main(void){

	double time_spent = 0.0;
	clock_t begin = clock();

	unsigned long long int i;
	int numprimes = 0;
	//for (i = TRILLION*1000; i < TRILLION*1000 + 10000; i++)
	for (i = 0; i < 1000; i++)
		if (trial_division(i)){
			numprimes ++;
			printf("%llu is prime \n", i);
		}



	clock_t end = clock();
	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Time elpased is %f seconds\n", time_spent);
	printf("In the 1000 trillion range average time %f\n", time_spent/10000);
	printf("number of primes in this range %d\n", numprimes);
}
