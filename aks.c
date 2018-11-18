#include <stdio.h>
#include <stdlib.h>
#define TRILLION 1000000000000
#define MILLION 1000000
#include <time.h> 
#include <math.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

long long combi(int n,int k)
{
    long long ans=1;
    k=k>n-k?n-k:k;
    int j=1;
    for(;j<=k;j++,n--)
    {
        if(n%j==0)
        {
            ans*=n/j;
        }else
        if(ans%j==0)
        {
            ans=ans/j*n;
        }else
        {
            ans=(ans*n)/j;
        }
    }
    //printf("ret %llu \n", ans);
    return ans;
}




int aks(unsigned long long int n){
	printf("%llu n \n", n);
	if (n <= 1)
		return 0;

	// step 1 check if perfect power
	int b; double a;
	for (b = 2; b <= log(n)/log(2); b++){
		//printf("%d b \n", b);
		a = pow(n, 1/(double)b);
		//printf("%f a \n", a);
		if (a == floor(a))
			return 0;
	}

	// step 2
	// Find the smallest r such that Or(n) > (log2 n)^2
	double maxk = pow((log(n)/log(2)), 2);
	double maxr = MAX(3, pow((log(n)/log(2)), 5));
	//printf("maxk %f \n", maxk);
	//printf("maxr %f \n", maxr);
	int nextR = 1;
	int r, k;
	for (r = 2; (nextR && (r<maxr)); r++){
		//printf("r %d \n", r);
		nextR = 0;
		for (k = 1; ((!nextR)&& (k <= maxk)); k ++){
			//printf("k %d \n", k);
			//nextR = (((int_pow(n,k) % r) == 1)  ||  ((int_pow(n,k) % r) == 0));
			nextR = ((power_mod(n,k,r) == 1)  ||  (power_mod(n,k,r) == 0));
		}
	}
	r --;

	//printf("r %d \n", r);

	// step 3
	// If 1 < gcd(a,n) < n for some a ≤ r, output composite.
	int i;
	for (i = r; i > 1; i --){
		int g = gcd(i, n);
		if ((g > 1) && (g < n))
			return 0;
	}

	// step 4
	// If n ≤ r, output prime.
	if (n <= r)
		return 1;

	// step 5
	// For a = 1 to sqrt(euler(r)log2(n))
	// if (X+a)n≠ Xn+a (mod Xr − 1,n), output composite;
	int max = floor((log(n)/log(2)) * sqrt(eulerPhi(r)));

	int diff = n-r;
	int A[diff];
	for (i = 0; i < diff; i++){
		A[i] = (combi(n,n-i) % n);
		printf("a %d \n", A[i]);
	} 

	long long int sum = 0;
	for (a=1; a <= max; a++){
		//if(polyModuloTest(a, n, r))
			//return 0;
		sum = 0;
		for (i = 0; i < diff; i++){
			sum += (A[i] * int_pow(a,i));
		}
		sum += int_pow(a,n);
		sum -= int_pow()

	}

	// step 6
	return 1;


}



// general helpers 
int int_pow(int base, int exp)
{
    int result = 1;
    while (exp)
    {
        if (exp & 1)
           result *= base;
        exp /= 2;
        base *= base;
    }
    return result;
}

int power_mod(int x, unsigned int y, int p) 
{ 
    int res = 1;      // Initialize result 
  
    x = x % p;  // Update x if it is more than or  
                // equal to p 
  
    while (y > 0) 
    { 
        // If y is odd, multiply x with result 
        if (y & 1) 
            res = (res*x) % p; 
  
        // y must be even now 
        y = y>>1; // y = y/2 
        x = (x*x) % p;   
    } 
    return res; 
} 














int gcd(int a, int b)
{
    int temp;
    while (b != 0)
    {
        temp = a % b;

        a = b;
        b = temp;
    }
    return a;
}

// step 5 helpers
int eulerPhi(int n){
	int result = 0;
    int k;
    for(k = 1; k <= n; k++)
        result += gcd(k, n) == 1;
    return result;
}


int polyModuloTest(int a, int n, int r){
	//int* polya = malloc(sizeof(int)*n);
	int* polyb = malloc(sizeof(int)*n);



	// for ()
	// 	if()
	// 		return 0;

	// free(polya);
	// free(polyb);

}





int main(void){

	double time_spent = 0.0;
	clock_t begin = clock();

	unsigned long long int i;
	long int numprimes = 0;
	aks(31);
	// //for (i = TRILLION*1000; i < TRILLION*1000 + 10000; i++)
	// for (i = 0; i < 10; i++)
	// 	if (aks(i)){
	// 		numprimes ++;
	// 		printf("%llu is prime \n", i);
	// 	}



	clock_t end = clock();
	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Time elpased is %f seconds\n", time_spent);
	printf("In the 1000 trillion range average time %f\n", time_spent/10000);

}




