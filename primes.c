#include <stdio.h>
#include <stdlib.h>
#define TRILLION 1000000000000
#define MILLION 1000000
#include <time.h> 

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
	unsigned long long int i = 5;
	while (i*i <= n){
		if (((n % i) == 0) || ((n % (i+2)) == 0))
			return 0;
		i = i + 6;
	}
	return 1;

}


//*************************************
//
//			Main
//
//*************************************

int main(void){

	double time_spent = 0.0;
	clock_t begin = clock();

	unsigned long long int i;
	long int numprimes = 0;
	for (i = TRILLION*1000; i < TRILLION*1000 + 10000; i++)
		if (trial_division(i)){
			numprimes ++;
			//printf("%llu is prime \n", i);
		}



	clock_t end = clock();
	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Time elpased is %f seconds\n", time_spent);
	printf("In the 1000 trillion range average time %f\n", time_spent/10000);

}