/**********************************************************************

 backwardアルゴリズム

  HMMモデルλが与えられたとき観測系列Oの確率 P(O|λ)を計算する。

  使用法: drillBWD


  初版                                         2011年04月27日 高木一幸
  実習用にコードを削除して穴埋め部分作成       2011年05月17日 高木一幸
  2012年度用に穴埋め部分を調整　　　　　       2012年05月01日 高木一幸
  総合情報学科専門実験用のcodeを制作           2012年10月03日 高木一幸
  新年度用にパラメータを変更                   2013年10月20日 高木一幸
  新年度用にパラメータを変更                   2014年10月09日 高木一幸
  新年度用にパラメータを変更、
                   backwardとmainを分離        2023年10月09日 高木一幸

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
#define NS              4                /* HMMの状態数              */
#define MS              2                /* 出力シンボル種類数       */
#define TLEN            4                /* 観測シンボル長("0110"  ) */



/*---------------------------------------------------------------------

 function prototypes

---------------------------------------------------------------------*/
double backward(unsigned int N,           /* HMMの状態数             */
	        unsigned int M,           /* 出力シンボル種類数      */
	        unsigned int T,           /* 観測シンボル長          */ 
		int *O,                   /* 観測記号列              */
		double **a,               /* HMM状態遷移確率         */
		double **b,               /* HMMシンボル出力確率     */
		double *pi,               /* HMM初期状態確率         */
		double **beta             /* backward変数β           */
		);



/*---------------------------------------------------------------------


 main


---------------------------------------------------------------------*/
int main(int argc,char *argv[])
{
  int i, j;                              /* 状態index                */   
  int t;                                 /* 時間index                */
  double **a;                            /* HMM状態遷移確率 [N*N]    */
  double **b;                            /* HMMシンボル出力確率 [N*M]*/
  double *pi;                            /* HMM初期状態確率 [N]      */
  double **beta;                         /* backward変数β            */
  FILE *fp;                              /* HMMパラメータファイル    */
  int O[TLEN] = {0,1,1,0};               /* 観測シンボル "0110"      */
  double prob;                           /* backward確率             */


  /*------------------------------------
    配列を生成
  ------------------------------------*/
  if(NULL==(a = (double**)calloc(NS,sizeof(double*)))) {
    fprintf(stderr,"%s: error in allocation of a\n",argv[0]);
    exit(EXIT_FAILURE);
  }
  for(i=0; i<NS; i++) {
    if(NULL==(a[i] = (double*)calloc(NS,sizeof(double)))) {
      fprintf(stderr,"%s: error in allocation of a[%d]\n",argv[0],i);
      exit(EXIT_FAILURE);
    }
  }
  if(NULL==(b = (double**)calloc(NS,sizeof(double*)))) {
    fprintf(stderr,"%s: error in allocation of b\n",argv[0]);
    exit(EXIT_FAILURE);
  }
  for(i=0; i<NS; i++) {
    if(NULL==(b[i] = (double*)calloc(MS,sizeof(double)))) {
      fprintf(stderr,"%s: error in allocation of b[%d]\n",argv[0],i);
      exit(EXIT_FAILURE);
    }
  }
  if(NULL==(pi = (double*)calloc(NS,sizeof(double)))) {
    fprintf(stderr,"%s: error in allocation of pi\n",argv[0]);
    exit(EXIT_FAILURE);
  }
  if(NULL==(beta = (double**)calloc(TLEN,sizeof(double*)))) {
    fprintf(stderr,"%s: error in allocation of beta\n",argv[0]);
    exit(EXIT_FAILURE);
  }
  for(t=0; t<TLEN; t++) {
    if(NULL==(beta[t] = (double*)calloc(NS,sizeof(double)))) {
      fprintf(stderr,"%s: error in allocation of beta[%d]\n",argv[0],t);
      exit(EXIT_FAILURE);
    }
  }

  /*------------------------------------
    HMMパラメータを読み込む
  ------------------------------------*/
  if(NULL==(fp = fopen("./paramFB.txt","r"))) {
    fprintf(stderr,"%s: error in opening parameter file ./paramFB.txt.\n",argv[0]);
    exit(EXIT_FAILURE);
  }
  for(i=0; i<NS; i++) {
    for(j=0; j<NS; j++) {
      fscanf(fp,"%lf", &a[i][j]);
    }
  }
  for(i=0; i<NS; i++) {
    for(j=0; j<MS; j++) {
      fscanf(fp,"%lf", &b[i][j]);
    }
  }
  for(i=0; i<NS; i++) {
    fscanf(fp,"%lf", &pi[i]);
  }
  fclose(fp);

  /*------------------------------------
     backward変数βを計算
  ------------------------------------*/
  prob = backward(NS,MS,TLEN,O,a,b,pi,beta);

  /*------------------------------------
    βの値をプリント
  ------------------------------------*/
  for(i=NS-1; i>=0; i--) {
    for(t=0; t<TLEN; t++) {
      printf("%10.8f\t",beta[t][i]);
    }
    printf("\n");
  }

  /*------------------------------------
     backward確率をプリント
  ------------------------------------*/
  printf("P(O|λ)= %10.8f\n",prob);

}
