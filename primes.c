#include <stdio.h>
#include <stdlib.h>

int trial_division(unsigned long long int n){
	if (n <= 1)
		return 0;
	else if (n <= 3)
		return 1;
	else if (((n % 2) == 0) || ((n % 3) == 0))
		return 0;
	int i = 5;
	while (i*i <= n){
		if (((n % i) == 0) || ((n % (i+2)) == 0))
			return 0;
		i = i + 6;
	}
	return 1;

}




int main(void){
	int i;
	for (i = 1000; i < 10000; i++)
		if (trial_division(i))
			printf("%d is prime \n", i);
}