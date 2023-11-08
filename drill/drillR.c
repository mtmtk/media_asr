/**********************************************************************

  総合情報学実験メディア「音声認識」

  drillR: HMMによるパターン認識実験

    使用法: drillR hmm1.txt hmm2.txt datalist

        入力    datalist: 認識用データファイル一覧ファイル


                                               2023年10月09日 高木一幸

***********************************************************************/
/*---------------------------------------------------------------------

 include files

---------------------------------------------------------------------*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <values.h>
#include <string.h>



/*---------------------------------------------------------------------

 macros

---------------------------------------------------------------------*/
#define NHMM            2                /* HMMの種類数              */
#define NS              2                /* HMMの状態数              */
#define MS              2                /* 出力シンボル種類数       */
#define TLEN            50               /* 観測シンボル長T          */
#define NDATA           100              /* 認識用データ数           */
#define FNAMELEN        256         /* パス名,ファイル名の最大文字数 */
#define TXTLINELEN       32  /* HMMパラメータファイル1行の最大文字数 */



/*---------------------------------------------------------------------

 function prototypes

---------------------------------------------------------------------*/
double forward(unsigned int N,          /* HMMの状態数               */
	       unsigned int M,          /* 出力シンボル種類数        */
	       unsigned int T,          /* 観測シンボル長            */ 
	       int *O,                  /* 観測記号列                */
	       double **a,              /* HMM状態遷移確率 [N*N]     */
	       double **b,              /* HMMシンボル出力確率 [N*M] */
	       double *pi,              /* HMM初期状態確率 [N]       */
	       double **alpha           /* forward変数α [T*N]        */
	       );



/*----------------------------------------------------------------------


  main


----------------------------------------------------------------------*/
int main(int argc, char	*argv[])
{
  int h, i, j, k;
  int t;
  double ***a;                           /* HMM状態遷移確率 [N*N]      */
  double ***b;                           /* HMMシンボル出力確率 [N*M]  */
  double *pi;                            /* HMM初期状態確率 [N]        */
  double **alpha;                        /* forward変数α               */
  int O[TLEN];                            /* 観測データ                */
  char fname_list[FNAMELEN];             /* データファイル一覧ファイル */
  FILE *fp_list;                 /* データファイル一覧ファイルハンドル */
  char fname_data[FNAMELEN];               /* データファイル名         */
  FILE *fp_hmm;                            /* HMMパラメータファイル    */
  char linebuf[TXTLINELEN]; /* HMMパラメータファイル1行を読み込むテキストバッファ */
  FILE *fp_data;                           /* データファイル           */
  int i_data;                             /* データindex               */
  double prob0, prob1;                    /* HMM1とHMM2の尤度          */
  int n0;                             /* カテゴリ1と認識されたデータ数 */
  int n1;                             /* カテゴリ2と認識されたデータ数 */



  /*------------------------------------
   コマンドライン引数の処理
  ------------------------------------*/
  if(4 == argc) {
    strcpy(fname_list,argv[3]);
  }
  else {
    fprintf(stderr,"usage: drillR hmm1.txt hmm2.txt datalist\n");
    fprintf(stderr,"       [IN] hmm1.txt: parameter file of HMM1\n");
    fprintf(stderr,"            hmm2.txt: parameter file of HMM2\n");
    fprintf(stderr,"            datalist: recognition data list\n");
    exit (EXIT_FAILURE);
  }

  
  /*------------------------------------
    配列を生成
  ------------------------------------*/
  if(NULL==(a =(double***)calloc(NHMM,sizeof(double**)))) {
     fprintf(stderr,"%s: error in allocation of a\n",argv[0]);
    exit(EXIT_FAILURE);
  }
  for(h=0; h<NHMM; h++) {
    if(NULL==(a[h] = (double**)calloc(NS,sizeof(double*)))) {
      fprintf(stderr,"%s: error in allocation of a[%d]\n",argv[0],h);
      exit(EXIT_FAILURE);
    }
    for(i=0; i<NS; i++) {
      if(NULL==(a[h][i] = (double*)calloc(NS,sizeof(double)))) {
	fprintf(stderr,"%s: error in allocation of a[%d][%d]\n",argv[0],h,i);
	exit(EXIT_FAILURE);
      }
    }
  }
  
  if(NULL==(b = (double***)calloc(NHMM,sizeof(double**)))) {
    fprintf(stderr,"%s: error in allocation of b\n",argv[0]);
    exit(EXIT_FAILURE);
  }
  for(h=0; h<NHMM; h++) {
    if(NULL==(b[h] = (double**)calloc(NS,sizeof(double)))) {
      fprintf(stderr,"%s: error in allocation of b[%d]\n",argv[0],h);
      exit(EXIT_FAILURE);
    }
    for(i=0; i<NS; i++) {
      if(NULL==(b[h][i] = (double*)calloc(MS,sizeof(double)))) {
	fprintf(stderr,"%s: error in allocation of b[%d][%d]\n",argv[0],h,i);
	exit(EXIT_FAILURE);
      }
    }
  }
  if(NULL==(pi = (double*)calloc(NS,sizeof(double)))) {
    fprintf(stderr,"%s: error in allocation of pi\n",argv[0]);
    exit(EXIT_FAILURE);
  }
  if(NULL==(alpha = (double**)calloc(TLEN,sizeof(double)))) {
    fprintf(stderr,"%s: error in allocation of alpha\n",argv[0]);
    exit(EXIT_FAILURE);
  }
  for(t=0; t<TLEN; t++) {
    if(NULL==(alpha[t] = (double*)calloc(NS,sizeof(double)))) {
      fprintf(stderr,"%s: error in allocation of alpha[%d]\n",argv[0],t);
      exit(EXIT_FAILURE);
    }
  }

  /*------------------------------------
    HMMパラメータをファイルから読み込む
  ------------------------------------*/
  for(i=0; i<NS; i++) {
    pi[i] = 0==i ? 1.0 : 0.0;
  }
  for(h=0; h<NHMM; h++) {
    if(NULL==(fp_hmm = fopen(argv[h+1],"rt"))) {
      fprintf(stderr,"%s: error in opening %s\n",argv[0],argv[h+1]);
      exit(EXIT_FAILURE);
    }
    if(NULL==(fgets(linebuf,TXTLINELEN,fp_hmm))) {
      fprintf(stderr,"%s: error in fgets on %s\n",argv[0],argv[h+1]);
      exit(EXIT_FAILURE);
    }
    if(NULL==(fgets(linebuf,TXTLINELEN,fp_hmm))) {
      fprintf(stderr,"%s: error in fgets on %s\n",argv[0],argv[h+1]);
      exit(EXIT_FAILURE);
    }
    if(NULL==(fgets(linebuf,TXTLINELEN,fp_hmm))) {
      fprintf(stderr,"%s: error in fgets on %s\n",argv[0],argv[h+1]);
      exit(EXIT_FAILURE);
    }
    for(i=0; i<NS; i++) {
      for(j=0; j<NS; j++) {
	if(1 != (fscanf(fp_hmm,"%lf",&a[h][i][j]))) {
	  fprintf(stderr,"%s: error in fscan(a[%d][%d][%d]) on %s\n",argv[0],h,i,j,argv[h+1]);
	  exit(EXIT_FAILURE);
	}
      }
    }
    if(NULL==(fgets(linebuf,TXTLINELEN,fp_hmm))) {
      fprintf(stderr,"%s: error in fgets on %s\n",argv[0],argv[h+1]);
      exit(EXIT_FAILURE);
    }    if(NULL==(fgets(linebuf,TXTLINELEN,fp_hmm))) {
      fprintf(stderr,"%s: error in fgets on %s\n",argv[0],argv[h+1]);
      exit(EXIT_FAILURE);
    }
    for(j=0; j<NS; j++) {
      for(k=0; k<MS; k++) {
	if(1 != (fscanf(fp_hmm,"%lf",&b[h][j][k]))) {
	  fprintf(stderr,"%s: error in fscan(b[%d][%d][%d]) on %s\n",argv[0],h,i,j,argv[h+1]);
	  exit(EXIT_FAILURE);
	}
      }
    }
    fclose(fp_hmm);
  }

  /*-------------------------------------
     学習データファイルの一覧ファイルを開く
  -------------------------------------*/
  if(NULL==(fp_list=fopen(fname_list,"r"))) {
    fprintf(stderr,"%s: error in opening %s\n",argv[0],fname_list);
    exit(EXIT_FAILURE);
  }

  /*-------------------------------------
     認識
  -------------------------------------*/
  n0 = n1 = 0;
  for(i_data=0; i_data<NDATA; i_data++) {
    /* データの読み込み */
    if(NULL==(fgets(fname_data,FNAMELEN,fp_list))) {
      fprintf(stderr,"%s: error in reading %s\n",argv[0],fname_list);
      exit(EXIT_FAILURE);
    }
    fname_data[strlen(fname_data)-1] = '\0';
    if(NULL==(fp_data=fopen(fname_data,"r"))) {
      fprintf(stderr,"%s: error in opening %s\n",argv[0],fname_data);
      exit(EXIT_FAILURE);
    }
    for(t=0; t<TLEN; t++) {
      fscanf(fp_data,"%d",O+t);
    }
    /* forward確率を計算し表示 */
    prob0 = log(forward(NS,MS,TLEN,O,a[0],b[0],pi,alpha)); /* HMM1 */
    prob1 = log(forward(NS,MS,TLEN,O,a[1],b[1],pi,alpha)); /* HMM2 */
    printf("%f\t%f\n",prob0,prob1);

    /* 各カテゴリに認識されたデータ数を集計 */
    if(prob0 >= prob1) n0++;
    if(prob0 < prob1) n1++;

    fclose(fp_data);
  }

  /* 各カテゴリに認識されたデータ数を表示 */
  fprintf(stderr,"n1= %d\n",n0);
  fprintf(stderr,"n2= %d\n",n1);

  fclose(fp_list);

  exit(0);
}
