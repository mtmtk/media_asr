/**********************************************************************

  drillT: BaumWelchテストプログラム

    使用法: drillT tdata.txt param.txt

    [IN]   tdata.txt: training data
    [OUT]  param.txt

***********************************************************************/
/*---------------------------------------------------------------------

 include files

---------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>



/*---------------------------------------------------------------------

 macros

---------------------------------------------------------------------*/
#define NS               2                /* HMMの状態数             */
#define MS               2                /* 出力シンボル種類数      */
#define TLEN             1000             /* 観測シンボル長T         */



/*----------------------------------------------------------------------

 function prototype

----------------------------------------------------------------------*/
int baumWelch(unsigned int N,            /* HMMの状態数               */
	      unsigned int M,            /* 出力シンボル種類数        */
	      unsigned int T,            /* 観測シンボル長            */ 
	      int *O,                    /* 観測記号列                */
	      double **a,                /* HMM状態遷移確率           */
	      double **b,                /* HMMシンボル出力確率       */
	      double *pi
	      );

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

void printHMM(FILE *fp,     /* 学習結果のパラメータを保存するファイル */
	      double **a,                 /* HMM状態遷移確率          */
	      double **b,                 /* HMMシンボル出力確率      */
	      double *pi                  /* HMM初期状態確率          */
	      );





/*----------------------------------------------------------------------


  main


----------------------------------------------------------------------*/
int main(int argc, char	*argv[])
{
  int i, j, k;                           /* 状態index                 */
  int t;                                 /* 時間index                 */
  double **a;                            /* HMM状態遷移確率 [N*N]     */
  double **b;                            /* HMMシンボル出力確率 [N*M] */
  double *pi;                            /* HMM初期状態確率 [N]       */
  double **alpha;                        /* forward変数α              */
  double **beta;                         /* backward変数β             */
  FILE *fp_tdata, *fp_param;     /* 学習データ、HMMパラメータファイル */
  int *O;                                /* 観測シンボル              */
 

  /*------------------------------------
    コマンドライン引数の数を検査
  ------------------------------------*/
  if(3 != argc) {
    fprintf(stderr,"usage: drillT tdata.txt param.txt\n");
    fprintf(stderr,"       [IN]   tdata.txt: training data\n");
    fprintf(stderr,"       [OUT]  param.txt\n");
    exit(EXIT_FAILURE);
  }

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
  if(NULL==(O = (int*)calloc(TLEN,sizeof(int)))) {
    fprintf(stderr,"%s: error in allocation of O\n",argv[0]);
    exit(EXIT_FAILURE);
  }

  /*------------------------------------
    パラメータの初期値を設定
  ------------------------------------*/
  for(i=0; i<NS; i++) {
    pi[i] = 0==i ? 1.0 : 0.0;
    for(j=0; j<NS; j++) {
      a[i][j] = 1.0/NS;
    }
    for(k=0; k<MS; k++) {
      b[i][k] = 1.0/MS;
    }
  }

  /*------------------------------------
    学習データを読み込む
  ------------------------------------*/
  if(NULL==(fp_tdata = fopen(argv[1],"rt"))) {
    fprintf(stderr,"%s: error in opening training data file %s.\n",argv[0],argv[1]);
    exit(EXIT_FAILURE);
  }
  for(t=0; t<TLEN; t++) {
    fscanf(fp_tdata,"%d",O+t);
  }
  fclose(fp_tdata);
  
  /*------------------------------------
    HMMパラメータを表示
  ------------------------------------*/
  fprintf(stderr,"Before Training ---------------\n");
  printHMM(stderr,a,b,pi);
  fprintf(stderr,"\n");

  /*------------------------------------
    BaumWelchアルゴリズムを実行
  ------------------------------------*/
  (void)baumWelch(NS,MS,TLEN,O,a,b,pi);
  
  /*------------------------------------
    HMMパラメータを表示
  ------------------------------------*/
  fprintf(stderr,"After Training ----------------\n");
  printHMM(stderr,a,b,pi);

  /*------------------------------------
    HMMパラメータをファイルに保存
  ------------------------------------*/
  if(NULL==(fp_param = fopen(argv[2],"wt"))) {
    fprintf(stderr,"%s: error in opening parameter file %s.\n",argv[0],argv[2]);
    exit(EXIT_FAILURE);
  }
  printHMM(fp_param,a,b,pi);
  fclose(fp_param);
  
  exit(EXIT_SUCCESS);
}



/*-----------------------------------------------------------------------

 printHMM: HMMパラメータを表示

-----------------------------------------------------------------------*/
void printHMM(FILE *fp,           /* 学習結果のパラメータを保存するファイル */
	      double **a,                  /* HMM状態遷移確率           */
	      double **b,                  /* HMMシンボル出力確率        */
	      double *pi                   /* HMM初期状態確率           */
	      )
{
  int i, j, k;


  fprintf(fp,"pi=\n");
  for(i=0; i<NS; i++) {
    fprintf(fp,"%f ",pi[i]);
  }
  fprintf(fp,"\n");
  
  fprintf(fp,"A=\n");
  for(i=0; i<NS; i++) {
    for(j=0; j<NS; j++) {
      fprintf(fp,"%f ",a[i][j]);
    }
    fprintf(fp,"\n");
  }

  fprintf(fp,"B=\n");
  for(i=0; i<NS; i++) {
    for(k=0; k<MS; k++) {
      fprintf(fp,"%f ",b[i][k]);
    }
    fprintf(fp,"\n");
  }
}
