/**********************************************************************

  drillG: 多次元ガウス混合密度関数のテスト

  使用法: drillG param.hmm o.mfcc
             param.hmm: HMMパラメータファイル
                o.mfcc: テスト用MFCC時系列

  初版                                          2023年10月11日 高木一幸

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
#define MINUSINF        -1.0e+30        /* 負の無限大                */
#define MAXFNAMELEN     512         /* パス名,ファイル名の最大文字数 */
#define WORDNAMELEN     64               /* 単語名の文字数           */



/*---------------------------------------------------------------------

 data definitions

---------------------------------------------------------------------*/
/* HTK形式ヘッダ */
typedef struct htktype{
  int   nSamples;                        /* サンプル数               */
  int   sampPeriod;                      /* サンプル間隔             */
  short sampSize;                        /* サンプルサイズ[byte]     */
  short parmKind;                        /* パラメータの種類         */
} HTKHEADER;          

/* HMMパラメータ */
typedef	struct{
  float *a;                              /* 状態遷移確率             */
  float *c;                              /* 混合係数                */
  float **mu;                            /* 平均                   */
  float **sigma2;                        /* 分散                   */
} HMM;



/*---------------------------------------------------------------------

 function prototypes

---------------------------------------------------------------------*/
double viterbi(int n_frame,              /* フレーム数               */
	       float **fv,               /* 特徴ベクトル             */
	       int n_dim,                /* 特徴ベクトル次元数       */
	       int n_state,              /* HMMの状態数              */
	       int n_mix,                /* Gaussian混合数           */
	       HMM *hmm,                 /* 単語HMM                  */
	       int *bestPath,            /* 最適状態系列             */
	       char *progname);    /* このプログラムの実行ファイル名 */

int argmax(double x[],
	   int n);



/*---------------------------------------------------------------------


 main


---------------------------------------------------------------------*/
int main(int argc,char *argv[])
{
  char progname[MAXFNAMELEN];      /* このプログラムの実行ファイル名 */
  int n_dim;                              /* 特徴ベクトル次元数       */
  int n_state;                           /* HMMの状態数              */
  int i_state, j_state;                  /* HMMの状態カウンタ        */
  int n_mix;                             /* HMMのGaussian混合数      */
  int i_mix;                            /* HMMのGaussian混合カウンタ */
  HMM *hmm;                               /* 単語HMM                 */
  char fname_HMM[MAXFNAMELEN];           /* 単語HMMファイル名        */
  FILE *fp_HMM;                          /* 単語HMMファイル          */
  char fname_mfcc[MAXFNAMELEN];       /* テスト用MFCC時系列ファイル名 */
  FILE *fp_mfcc;                        /* テスト用MFCC時系列ファイル */
  HTKHEADER htk;                         /* HTK形式ヘッダ            */
  float **fv;                /* 特徴ベクトル保存用配列(#frame * dim) */
  int n_dim_input;                     /* 入力音声特徴ベクトル次元数 */
  int i_dim;                             /* 特徴ベクトル次元カウンタ */
  int n_frame;                           /* 入力音声フレーム数       */
  int i_frame;                         /* 入力音声フレーム数カウンタ */
  char linebuff[MAXFNAMELEN]; /*テキストファイル読み込み用1行バッファ*/
  double lh;                            /* 尤度                     */
  int *bestPath;                        /* 最尤状態系列             */
  int result;                           /* 認識結果                 */


  /*------------------------------------------------------------
   コマンドライン数の検査
  ------------------------------------------------------------*/
  (void)strcpy(progname,argv[0]);
  if(3 != argc) {
    fprintf(stderr,"使用法: drillG param.hmm o.mfcc\n");
    fprintf(stderr,"              param.hmm: HMMパラメータファイル\n");
    fprintf(stderr,"                 o.mfcc: テスト用MFCC時系列\n");
    exit(EXIT_FAILURE);
  }
  (void)strcpy(fname_HMM,argv[1]);
  (void)strcpy(fname_mfcc,argv[2]);

 
  /*--------------------------------------
    HMMパラメータの読み込み
  --------------------------------------*/
  n_dim = 20;
  n_state = 5;
  n_mix = 4;
  /* HMMパラメータ配列の生成 */
  /* 単語のHMM状態 */
  if(NULL==(hmm=(HMM*)calloc(n_state,sizeof(HMM)))) {
      fprintf(stderr,"%s: error in allocating hmm\n",progname);
      exit(EXIT_FAILURE);
  }
  for(i_state=0; i_state<n_state; i_state++) {
    /* 状態遷移確率 a_{ij} */
    if(NULL == (hmm[i_state].a=(float*)calloc(2,sizeof(float)))) {
      fprintf(stderr,"%s: error in allocating hmm[%d].a\n",progname,i_state);
      exit(EXIT_FAILURE);
    }
    if(NULL == (hmm[i_state].c=(float*)calloc(n_mix,sizeof(float)))){
      fprintf(stderr,"%s: error in allocating hmm[%d].c\n",progname,i_state);
      exit(EXIT_FAILURE);
    }
    /* 平均値μ */
    if(NULL == (hmm[i_state].mu=(float**)calloc(n_mix,sizeof(float*)))){
      fprintf(stderr,"%s: error in allocating hmm[%d].mu\n",progname,i_state);
      exit(EXIT_FAILURE);
    }
    for(i_mix=0; i_mix<n_mix; i_mix++) {
      if(NULL == (hmm[i_state].mu[i_mix]=(float*)calloc(n_dim,sizeof(float)))){
	fprintf(stderr,"%s: error in allocating hmm[%d].mu[%d]\n",progname,i_state,i_mix);
	exit(EXIT_FAILURE);
      }
    }
    /* 分散値 σ^2 */
    if(NULL == (hmm[i_state].sigma2=(float**)calloc(n_mix,sizeof(float*)))){
      fprintf(stderr,"%s: error in allocating hmm[%d].sigma2\n",progname,i_state);
      exit(EXIT_FAILURE);
    }
    for(i_mix=0; i_mix<n_mix; i_mix++) {
      if(NULL == (hmm[i_state].sigma2[i_mix]=(float*)calloc(n_dim,sizeof(float)))){
	fprintf(stderr,"%s: error in allocating hmm[%d].sigma2[%d]\n",progname,i_state,i_mix);
	exit(1);
      }
    }
  }

  /*-------------------------------------------
    HMMパラメータの読み込み
  -------------------------------------------*/
  if(NULL==(fp_HMM=fopen(fname_HMM,"r"))) {
    fprintf(stderr,"%s: error in opening %s\n",progname,fname_HMM);
    exit(EXIT_FAILURE);
  }
  if(NULL==(fgets(linebuff,MAXFNAMELEN,fp_HMM))) {
    fprintf(stderr,"%s: error in fgets (dim) from %s\n",progname,fname_HMM);
    exit(EXIT_FAILURE);
  }
  if(NULL==(fgets(linebuff,MAXFNAMELEN,fp_HMM))) {
    fprintf(stderr,"%s: error in fgets (state) from %s\n",progname,fname_HMM);
    exit(EXIT_FAILURE);
  }
  if(NULL==(fgets(linebuff,MAXFNAMELEN,fp_HMM))) {
    fprintf(stderr,"%s: error in fgets (mix) from %s\n",progname,fname_HMM);
    exit(EXIT_FAILURE);
  }
  /* 状態遷移確率 */
  for(i_state=0; i_state<n_state; i_state++) {
    for(j_state=0; j_state<2; j_state++){
      fscanf(fp_HMM,"%e",&(hmm[i_state].a[j_state]));
    }
    /* 混合Gaussianパラメータ */
    for(i_mix=0; i_mix<n_mix; i_mix++) {
      /* 混合係数 */
      fscanf(fp_HMM,"%e",&(hmm[i_state].c[i_mix]));
      /* 平均 */
      for(i_dim=0; i_dim<n_dim; i_dim++) {
	fscanf(fp_HMM,"%e",&(hmm[i_state].mu[i_mix][i_dim]));
      }
      /* 分散 */
      for(i_dim=0; i_dim<n_dim; i_dim++) {
	fscanf(fp_HMM,"%e",&(hmm[i_state].sigma2[i_mix][i_dim]));
      }
    }
  }
  fclose(fp_HMM);

  /*-------------------------------------------
    テスト用MFCC時系列の読み込み
  -------------------------------------------*/
  if(NULL == (fp_mfcc = fopen(fname_mfcc,"r"))) {
    fprintf(stderr,"%s: error in opening %s\n",progname,fname_mfcc);
    exit(1);
  }
  /* HTK仕様ヘッダーの読み込み */
  if(1 != fread(&htk,sizeof(HTKHEADER),1,fp_mfcc)) {
    fprintf(stderr,"%s: error in reading HTK header from %s\n",progname,fname_mfcc);
    exit(1);
  }
  n_frame = htk.nSamples;
  n_dim_input = htk.sampSize/sizeof(float);

  /* テスト用MFCC時系列とHMMの次元の検査 */
  if(n_dim != n_dim_input) {
    fprintf(stderr,"%s: HMM dimension (%d) is not equal to input feature dimension (%d)\n",
	    progname,n_dim,n_dim_input);
    exit(EXIT_FAILURE);
  }

  /* 特徴ベクトル保存用配列の生成 */
  if(NULL==(fv=(float**)calloc(n_frame,sizeof(float*)))) {
    fprintf(stderr,"%s: error in allocating fv\n",progname);
    exit(EXIT_FAILURE);
  }
  for(i_frame=0; i_frame<n_frame; i_frame++) {
    if(NULL==(fv[i_frame]=(float*)calloc(n_dim_input,sizeof(float)))) {
      fprintf(stderr,"%s: error in allocating fv[%d]\n",progname,i_frame);
      exit(EXIT_FAILURE);
    }
  }
  /* 特徴量ベクトルの読み込み */
  for(i_frame=0; i_frame<n_frame; i_frame++) {
    if(n_dim_input != fread(fv[i_frame],sizeof(float),n_dim_input,fp_mfcc)) {
      fprintf(stderr,"%s: error in reading feature vector of frame %d from %s\n",
	      progname,i_frame,fname_mfcc);
      exit(EXIT_FAILURE);
    }
  }
  fclose(fp_mfcc);

  /* 最尤状態系列用配列を生成          */
  if(NULL==(bestPath=(int*)calloc(n_frame,sizeof(int)))) {
    fprintf(stderr,"%s: error in allocating bestPath\n",progname);
    exit(EXIT_FAILURE);
  }

  /* Viterbiデコーディング */
  lh = viterbi(n_frame,fv,n_dim,n_state,n_mix,hmm,bestPath,progname);

  /* 尤度の値を表示する */;
  printf("log likelihood= %e\n",lh);


  /*--------------------------------------
    HMM,計算用配列などの記憶領域を開放
  --------------------------------------*/
  for(i_state=0; i_state<n_state; i_state++) {
    free(hmm[i_state].a);
    free(hmm[i_state].c);
    for(i_mix=0; i_mix<n_mix; i_mix++) {
      free(hmm[i_state].mu[i_mix]);
    }
    free(hmm[i_state].mu);
    for(i_mix=0; i_mix<n_mix; i_mix++) {
      free(hmm[i_state].sigma2[i_mix]);
    }
    free(hmm[i_state].sigma2);
  }
  free(hmm);
  free(bestPath);
  for(i_frame=0; i_frame<n_frame; i_frame++) {
    free(fv[i_frame]);
  }
  free(fv);

  return(EXIT_SUCCESS);
}
