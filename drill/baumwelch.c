/**********************************************************************

  baumWelch.c: BaumWelchプログラム

***********************************************************************/
/*---------------------------------------------------------------------

 include files

---------------------------------------------------------------------*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <values.h>
#include <string.h>
#include <float.h>



/*---------------------------------------------------------------------

 macros

---------------------------------------------------------------------*/
#define EPSILON          1.0e-06          /* 再推定終了の閾値        */



/*----------------------------------------------------------------------

 function prototype

----------------------------------------------------------------------*/
double forward(unsigned int N,          /* HMMの状態数                */
	       unsigned int M,          /* 出力シンボル種類数         */
	       unsigned int T,          /* 観測シンボル長             */ 
	       int *O,                  /* 観測記号列                 */
	       double **a,              /* HMM状態遷移確率 [N*N]      */
	       double **b,              /* HMMシンボル出力確率 [N*M]  */
	       double *pi,              /* HMM初期状態確率 [N]        */
	       double **alpha           /* forward変数α [T*N]         */
	       );

double backward(unsigned int N,           /* HMMの状態数              */
	        unsigned int M,           /* 出力シンボル種類数       */
	        unsigned int T,           /* 観測シンボル長           */ 
		int *O,                   /* 観測記号列               */
		double **a,               /* HMM状態遷移確率          */
		double **b,               /* HMMシンボル出力確率      */
		double *pi,               /* HMM初期状態確率          */
		double **beta             /* backward変数β            */
		);



/*----------------------------------------------------------------------

 baumWelch

----------------------------------------------------------------------*/
int baumWelch(unsigned int N,            /* HMMの状態数               */
	      unsigned int M,            /* 出力シンボル種類数        */
	      unsigned int T,            /* 観測シンボル長            */ 
	      int *O,                    /* 観測記号列 [T]            */
	      double **a,                /* HMM状態遷移確率 [N*N]     */
	      double **b,                /* HMMシンボル出力確率 [N*M] */
	      double *pi                 /* HMM初期状態確率 [N]       */
	      )
{
  int t;
  int i, j, k;
  double **alpha;                        /* forward変数α              */
  double **beta;                         /* backward変数β             */
  int V[2] = {0,1};                      /* 出力シンボル              */
  double prob, prob_old=-DBL_MAX;        /* 尤度(パラメータ更新後,前) */
  double prob_forward, prob_backward;    /* forward, backward 確率    */
  double sum1,sum2;
  int n_reest = 0;                       /* 再推定回数                */


  /*------------------------------------
    配列を生成
  ------------------------------------*/
  if(NULL==(alpha= (double**)calloc(T,sizeof(double)))) {
    fprintf(stderr,"baumwelch: error in allocation of alpha\n");
    exit(EXIT_FAILURE);
  }
  for(t=0; t<T; t++) {
    if(NULL==(alpha[t] = (double*)calloc(N,sizeof(double)))) {
      fprintf(stderr,"baumwelch: error in allocation of alpha[%d]\n",t);
      exit(EXIT_FAILURE);
    }
  }
  if(NULL==(beta = (double**)calloc(T,sizeof(double)))) {
    fprintf(stderr,"baumwelch: error in allocation of beta\n");
    exit(EXIT_FAILURE);
  }
  for(t=0; t<T; t++) {
    if(NULL==(beta[t] = (double*)calloc(N,sizeof(double)))) {
      fprintf(stderr,"baumwelch: error in allocation of beta[%d]\n",t);
      exit(EXIT_FAILURE);
    }
  }

  for(;;) {
    /*------------------------------------
      forward確率αを計算
    ------------------------------------*/
    prob_forward = /* fill in blank */;
    prob = prob_forward;

    /*--------------------------------------
     更新したlogP(λ)と暫定parameter値を表示
    --------------------------------------*/
    printf("%d %e",n_reest,prob);
    for(i=0; i<N; i++) {
      for(j=0; j<N; j++) {
        printf(" %e",a[i][j]);
      }
    }
    for(i=0; i<N; i++) {
      for(k=0; k<M; k++) {
        printf(" %e",b[i][k]);
      }
    }
    printf("\n");

    /*------------------------------------
      収束したら再推定を終了
    ------------------------------------*/
    if(log(prob)-log(prob_old) < EPSILON) goto END;

    /* P(λ)の値を記憶 */
    prob_old = prob;

    /* 再推定回数をインクリメント */
    n_reest++;

    /*------------------------------------
      backward確率βを計算
    ------------------------------------*/
    prob_backward = /* fill in blank */;

    /*------------------------------------
      πi を再推定
    ------------------------------------*/
    for(i=0; i<N; i++)
      pi[i] = /* fill in blank */;

    /*------------------------------------
      A (a_ij) を再推定
    ------------------------------------*/
    for(i=0; i<N; i++) {
      for(j=0; j<N; j++) {
	sum1 = sum2 = 0.0;
        /* fill in blank */
        a[i][j] = sum1/sum2;
      }
    }

    /*------------------------------------
      B (b_jk) を再推定
    ------------------------------------*/
    for(j=0; j<N; j++) {
      for(k=0; k<M; k++) {
	sum1 = sum2 = 0;
        /* fill in blamk */
        b[j][k] = sum1/sum2;
      }
    }
  }
  
 END: fprintf(stderr,"Baum-Welch estimation converged. (n_reest= %d)\n",n_reest);
  
  return 0;
}
