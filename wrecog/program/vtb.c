/**********************************************************************

 viterbi: Viterbiアルゴリズム

   関数vtb等を別ファイルにする                     2013年10月07日 高木一幸

**********************************************************************/
/*---------------------------------------------------------------------

 include files

---------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>





/*---------------------------------------------------------------------

 macros

---------------------------------------------------------------------*/
#define MINUSINF        -1.0e+30         /* 負の無限大               */





/*---------------------------------------------------------------------

 data definitions

---------------------------------------------------------------------*/
/* HMMパラメータ */
typedef	struct{
  float *a;                              /* 状態遷移確率             */
  float *c;                              /* 混合係数                 */
  float **mu;                            /* 平均                     */
  float **sigma2;                        /* 分散                     */
} HMM;





/*---------------------------------------------------------------------

 function prototypes

---------------------------------------------------------------------*/
double gauss_mixr(int n_dim,
		  int n_mix,
		  int i_frame,
		  float **fv,
		  HMM hmm);

double gaussr(int n_dim,
	      int i_frame,
	      float **fv,
	      float *mu,
              float *sigma2);

double gpdf(int D,
	    float *o,
	    float *mu,
	    float *sigma2);



/*---------------------------------------------------------------------

 viterbi

---------------------------------------------------------------------*/
double viterbi(int n_frame,              /* フレーム数               */
	       float **fv,               /* 特徴ベクトル             */
	       int n_dim,                /* 特徴ベクトル次元数       */
	       int n_state,              /* HMMの状態数              */
	       int n_mix,                /* Gaussian混合数           */
	       HMM *hmm,                 /* 単語HMM                  */
	       int *bestPath,            /* 最適状態系列             */
	       char *progname)     /* このプログラムの実行ファイル名 */
{
  int i_word,j_word;                     /* 単語番号                 */
  int i_state;                           /* 状態番号                 */
  int i_frame,j_frame;                   /* フレーム番号             */
  double **delta;                        /* 単語モデルの累積尤度値   */
  double lh_loop, lh_next;               /* 尤度値                   */
  double LH;                             /* log P(O|λ)              */
  int **B;                               /* バックポインタ           */

  int i_mix, i_dim;



  /* 計算用配列δの生成 */
  if(NULL==(delta=(double**)calloc(n_frame,sizeof(double*)))) {
      fprintf(stderr,"%s: error in calloc delta in viterbi\n",progname);
      exit(EXIT_FAILURE);
  }
  for(i_frame=0; i_frame<n_frame; i_frame++) {
    if(NULL==(delta[i_frame]=(double*)calloc(n_state,sizeof(double)))) {
      fprintf(stderr,"%s: error in calloc delta[%d] in viterbi\n",progname,i_frame);
      exit(EXIT_FAILURE);
    }  
  }  

  /* バックポインタ用配列の生成 */
  if(NULL==(B=(int**)calloc(n_frame,sizeof(int*)))){
      fprintf(stderr,"%s: error in calloc B in viterbi\n",progname);
      exit(EXIT_FAILURE);
  }
  for(i_frame=0; i_frame<n_frame; i_frame++) {
    if(NULL==(B[i_frame]=(int*)calloc(n_state,sizeof(int)))){
      fprintf(stderr,"%s: error in calloc B[%d] in viterbi\n",progname,i_frame);
      exit(EXIT_FAILURE);
    }
  }

  /* 初期化 */
  delta[0][0] = log(gauss_mixr(n_dim,n_mix,0,fv,hmm[0]));
  B[0][0] = 0;
  for(i_state=1; i_state<n_state; i_state++) {
    delta[0][i_state] = MINUSINF;
    B[0][i_state] = 0;
  }


  /* 漸化式計算(t>0) */
  for(i_frame=1; i_frame<n_frame; i_frame++) {
    delta[i_frame][0] = delta[i_frame-1][0]
      +log(hmm[0].a[0])
      +log(gauss_mixr(n_dim,n_mix,i_frame,fv,hmm[0]));
    for(i_state=1; i_state<n_state; i_state++) {
       lh_next = delta[i_frame-1][i_state-1]+log(hmm[i_state].a[1]);
       lh_loop = delta[i_frame-1][i_state]+log(hmm[i_state].a[0]);
       if(lh_next > lh_loop) { /* 状態遷移した場合 */
	 delta[i_frame][i_state]
	   = delta[i_frame-1][i_state-1]
	   + log(hmm[i_state-1].a[1])
	   + log(gauss_mixr(n_dim,n_mix,i_frame,fv,hmm[i_state]));
	 B[i_frame][i_state] = i_state-1;
       }
       else { /* 状態loopした場合 */
	 delta[i_frame][i_state]
	   = delta[i_frame-1][i_state]
	   + log(hmm[i_state].a[0])
	   + log(gauss_mixr(n_dim,n_mix,i_frame,fv,hmm[i_state]));
	 B[i_frame][i_state] = i_state;
       }
    }
  }


  /* Viterbi確率 */
  LH = delta[n_frame-1][n_state-1];

  /* 最適状態系列の復元 */
  bestPath[n_frame-1] = B[n_frame-1][n_state-1];
  for(i_frame=n_frame-1; i_frame>1; i_frame--) {
    bestPath[i_frame-1] = B[i_frame][bestPath[i_frame]];
  }
 

  /* 計算用配列δの開放 */
  for(i_frame=0; i_frame<n_frame; i_frame++) {
    free(delta[i_frame]);
  }
  free(delta);

  /* バックポインタ用配列の解放 */
  for(i_frame=0; i_frame<n_frame; i_frame++) {
    free(B[i_frame]);
  }
  free(B);

  return LH;
}





/*---------------------------------------------------------------------

 gauss_mixr

---------------------------------------------------------------------*/
double gauss_mixr(int n_dim,
		  int n_mix,
		  int i_frame,
		  float **fv,
		  HMM hmm)
{
  int i_mix;
  double result;

  result = 0.0;
  for(i_mix=0; i_mix<n_mix; i_mix++) {
    result += hmm.c[i_mix]*gaussr(n_dim,i_frame,fv,hmm.mu[i_mix],hmm.sigma2[i_mix]);
  }

  return result;
}





/*---------------------------------------------------------------------

 gaussr

---------------------------------------------------------------------*/
double gaussr(int n_dim,
	      int i_frame,
	      float **fv,
	      float *mu,
	      float *sigma2)
{
  return gpdf(n_dim,fv[i_frame],mu,sigma2);
}
