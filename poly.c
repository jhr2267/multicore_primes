#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>

 
typedef struct {
        int power;
        double * coef;
} poly_t, *poly;
 
#define E(x, i) (x)->coef[i]
 
// /* passing in negative power to have a zeroed poly */
poly p_new(int power, ...)
{
        int i, zeroed = 0;
        va_list ap;
 
        if (power < 0) {
                power = -power;
                zeroed = 1;
        }
 
        poly p = malloc(sizeof(poly_t));
        p->power = power;
        p->coef = malloc(sizeof(double) * ++power);
 
        if (zeroed)
                for (i = 0; i < power; i++) p->coef[i] = 0;
        else {
                va_start(ap, power);
                for (i = 0; i < power; i++)
                        E(p, i) = va_arg(ap, double);
                va_end(ap);
        }
 
        return p;
}

poly p_new_x(int power, int x){
        poly p = malloc(sizeof(poly_t));
        p->power = power;
        p->coef = malloc(sizeof(double) * ++power);
        int i;
        for (i = 0; i < power; i++) p->coef[i] = 0;
        p->coef[0] = -1.;
        p->coef[x] = 1.;
        return p;
}


// // unused
// double binomial_coeff(int n, int k){
//         int i;
//         double ret = 1;
//         for (i = 1; i <= k; i++)
//                 ret *= (n+1-i)/i;
//         printf("ret %f \n", ret);
//         return ret;
// }


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




poly p_new_binomial(int power){
        poly p = malloc(sizeof(poly_t));
        p->power = power;
        p->coef = malloc(sizeof(double) * power+1);
        int i;
        for (i = 0; i <= power; i++)
                E(p, i) = (double)combi(power, i);

        return p;
}

poly p_new_binomial_a(int power, int a){
        poly p = malloc(sizeof(poly_t));
        p->power = power;
        p->coef = malloc(sizeof(double) * power+1);
        int i;
        for (i = 0; i <= power; i++)
                E(p, i) = (double)(combi(power, i) * pow(a,power-i));

        return p;
}







 
void p_del(poly p)
{
        free(p->coef);
        free(p);
}
 
void p_print(poly p)
{
        int i;
        for (i = 0; i <= p->power; i++)
                printf("%g ", E(p, i));
        printf("\n");
}
 
poly p_copy(poly p)
{
        poly q = p_new(-p->power);
        memcpy(q->coef, p->coef, sizeof(double) * (1 + p->power));
        return q;
}
 
/* p: poly;  d: divisor;  r: remainder; returns quotient */
poly p_div(poly p, poly d, poly* r)
{
        poly q;
        int i, j;
        int power = p->power - d->power;
        double ratio;
 
        if (power < 0) return 0;
 
        q = p_new(-power);
        *r= p_copy(p);
 
        for (i = p->power; i >= d->power; i--) {
                E(q, i - d->power) = ratio = E(*r, i) / E(d, d->power);
                E(*r ,i) = 0;
 
                for (j = 0; j < d->power; j++)
                        E(*r, i - d->power + j) -= E(d, j) * ratio;
        }
        while (! E(*r, --(*r)->power));
 
        return q;
}

void poly_mod(poly* p, int power, int m){
        int i;
        long long int b;
        for (i = 0; i <= power; i ++){
                b = E(*p ,i);
                printf("int %llu \n", b);
                printf("mod %llu \n", b % m);
                E(*p ,i) = b % m;
        }


}
 
int main()
{
        //poly p = p_new(8, 1., 2., 3., 4.);
        //poly p = p_new_binomial(31);
        poly p = p_new_binomial_a(31, 5);

        //poly d = p_new(31, -1.,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.);
        poly d = p_new_x(29, 29);

        // printf("poly: "); p_print(p);
        // printf("div:  "); p_print(d);

        poly r;
        poly q = p_div(p, d, &r);
 
        printf("poly: "); p_print(p);
        printf("div:  "); p_print(d);
        printf("quot: "); p_print(q);
        printf("rem:  "); p_print(r);

        poly_mod(&r, r->power, 31);

        printf("rem:  "); p_print(r);

 
        p_del(p);
        p_del(q);
        p_del(r);
        p_del(d);
 
        return 0;
}