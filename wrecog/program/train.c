/**********************************************************************

  train.c: HMM学習プログラム

  ※元のプログラムは ~takagi/edu/JJikken2013/program/trainHMM.c

  使用法:
      train trainingList n_state hmmFile logFile

  入力
          trainingList: 学習用MFCCファイルの一覧を書いたファイル
           n_state: HMMの状態数

  出力
           hmmFile: HMMのパラメータファイル
           logFile: 学習過程記録ファイル



                                                 2010年09月7日 高木一幸
  元のtrain.cの学習トークン一覧読み込み方式を変更(ファイル名指定に)
                                                2010年09月21日 高木一幸
  学習トークン一覧の行頭1文字が'#'の場合,読み飛ばすように変更
                                                2010年09月23日 高木一幸
  20次のMFCCで実行するよう変更                  2012年08月25日 高木一幸
  状態数,混合数はコマンドラインで指定,フレーム数最大値,トークン数は
  学習データに応じて決定,MFCC次元数はHTK形式MFCCヘッダから読み取る
  ように変更．                                  2013年09月04日 高木一幸
  ファイル名を "trainHMM.c" から "train.c" に変更
                                                2013年10月11日 高木一幸
  混合数は4に固定                                 2013年11月03日 高木一幸

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
#define	INFNTY          1.0e+06         /* 大きな値とみなす値        */
#define	EPSILON         1.0e-03         /* 再推定終了の閾値          */
#define	DELTA           1.0e-05         /* small number for centroid splitting */
#define	LAMBDA          1.0e-03         /* K平均法終了の閾値         */
#define	ZERO            1.0e-30         /* 零とみなす値              */
#define	FNAMELEN        256             /* length of input and output file name */





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
  float *c;                              /* 混合係数                 */
  float **mu;                            /* 平均                     */
  float **sigma2;                        /* 分散                     */
} HMM;





/*---------------------------------------------------------------------

 function prototypes

---------------------------------------------------------------------*/
void parm_init(int n_mix,                /* 混合数                   */
	       int n_dim,                /* 特徴ベクトル次元数       */
	       int n_token,              /* 学習データ数             */
	       int *n_frame,             /* 各トークンのフレーム数   */
	       int n_total_frame,        /* 学習データのフレーム総数 */
	       float ***fv,              /* 特徴ベクトル             */
	       int n_state,              /* HMM状態数                */
	       HMM *hmm);                /* HMMパラメータ            */

int lbg(int n_mix,                       /* 混合数                   */
	int n_dim,                       /* 特徴ベクトル次元数       */
	int n_token,                     /* 学習データ数             */
	float ***fv,                     /* 特徴ベクトル             */
	int *start,
	int *end,
	int **tokentable,
	int **frametable,
	int *numframe,
	float **centroid);

int kmeans(int n_mix,
           int n_dim,
	   int n_token,
	   float ***fv,
	   int *start,
	   int *end,
	   int n_total_frame,
	   int n_cluster,
	   int **tokentable,
	   int **frametable,
	   int *numframe,
	   float **centroid);

float euclid(float *v1,
	     float *v2,
	     int n_dim);

float re_estimate1(int n_mix,
		   int n_dim,
		   int n_token,
		   int *n_frame,
		   float ***fv,
		   int n_state,
		   HMM *hmm,
		   double ***alpha,
		   double ***beta,
		   double **scale,
		   double ****xi,
		   double ***gamma1,
		   double ****zeta);

int forward(int n_dim,
	    int i_token,
	    int *n_frame,
	    float ***fv,
	    int n_mix,
	    double ***alpha,
	    double **scale,
	    int n_state,
	    HMM *hmm);

int backward(int n_dim,
	     int i_token,
	     int *n_frame,
	     float ***fv,
	     int n_mix,
	     double ***beta,
	     double **scale,
	     int n_state,
	     HMM *hmm);

double gpdf(int D,
	    float *o,
	    float *mu,
	    float *sigma2);

double gauss_mix(int n_dim,
		 int n_mix,
		 int i_token,
		 int i_frame,
		 float ***fv,
		 float **mu,
		 float **sigma2,
		 float *c);

double gauss(int n_dim,
	     int i_frame,
	     float **fv,
	     float *mu,
	     float *sigma2);

int ilog2(int n);

int nan_inf_determin(float f);



/*---------------------------------------------------------------------

 global variables

---------------------------------------------------------------------*/
char progname[FNAMELEN];          /* このプログラムの実行ファイル名 */
FILE *fp_log;                           /* 学習記録ファイル         */



/*---------------------------------------------------------------------

 usage: 使用方法

---------------------------------------------------------------------*/
void usage() {
  fprintf(stderr, "\n");
  fprintf(stderr, " %s - HMMの学習\n",progname);
  fprintf(stderr, "\n");
  fprintf(stderr, "  使用法:\n");
  fprintf(stderr, "       %s trainingList n_state hmmFile logFile\n",progname);
  fprintf(stderr, "  入力:\n");
  fprintf(stderr, "     trainingList: 学習用MFCCファイルの一覧を書いたファイル\n");
  fprintf(stderr, "          n_state: HMMの状態数\n\n");
  fprintf(stderr, "出力\n");
  fprintf(stderr, "          hmmFile: HMMのパラメータファイル\n");
  fprintf(stderr, "          logFile: 学習過程記録ファイル\n");
  fprintf(stderr, "\n");
}





/*---------------------------------------------------------------------


  main


---------------------------------------------------------------------*/
int main(int argc, char	*argv[])
{
  /* コマンドラインパラメータ */
  char trainingList[FNAMELEN];   /* 学習データファイルの一覧ファイル */
  int n_state;                           /* HMMの状態数              */
  int n_mix;                             /* 混合ガウス分布の混合数   */
  char hmmFile[FNAMELEN];                /* HMMパラメータファイル    */
  char logFile[FNAMELEN];                /* 学習記録ファイル         */
  /* ファイル */
  FILE *fp_trainingList;         /* 学習データファイルの一覧ファイル */
  char fvFile[FNAMELEN];               /* 学習データファイル名       */
  FILE *fp_fv;                         /* 学習データファイル         */
  HTKHEADER htk;                       /* HTK形式ヘッダ              */
  FILE *fp_hmm;                        /* HMMパラメータファイル      */
  /* データ */
  int n_token;                         /* 学習データ数               */
  int *n_frame;                        /* 学習用データ毎のフレーム数 */
  int n_dim;                           /* 特徴ベクトルの次元数       */
  float ***fv;      /* 特徴ベクトル保存用配列(#token * #frame * dim) */
  int n_total_frame;                     /* 学習データの総フレーム数 */

  /* 配列のindex */
  int i_token;                           /* 学習データ               */
  int i_frame;                           /* フレーム                 */
  int i_dim;                             /* 特徴ベクトル要素         */
  int i_mix;                             /* 混合要素                 */
  int i_state, j_state;                  /* HMMの状態                */

  /* HMMパラメータ */
  HMM *hmm, *hmm_old;                    /* HMMパラメータ            */

  /* HMMパラメータ推定用計算配列 */
  double ***alpha;                       /* #token * #frame * #state */
  double ***beta;                        /* #token * #frame * #state */
  double **scale;                        /* #token * #frame          */
  double ****xi;                     /* #token * #frame * #state * 2 */
  double ***gamma1;                      /* #token * #frame * #state */
  double ****zeta;   /* NMAXTOKEN * MAXFRAME * NSTATE_P * GAUSS_MIX  */

  /* 再推定 */
  int reest_count;                       /* 再推定回数               */
  double best_min;
  int best_num;
  float logp_old, logp_new;              /* log probability           */


  /*------------------------------------
   コマンドラインパラメータの検査
  ------------------------------------*/
  strcpy(progname,argv[0]);
  if(5 != argc) {
    usage();
    exit(EXIT_FAILURE);
  }
  strcpy(trainingList,argv[1]);
  if( 0 >= (n_state = atoi(argv[2]))) {
    fprintf(stderr , "%s: invalid value for n_state (%d)\n",progname,n_state);
    exit(1);
  }
  n_mix = 4;
  strcpy(hmmFile,argv[3]);
  strcpy(logFile,argv[4]);


  /*------------------------------------
   ログファイルを開く
  ------------------------------------*/
  if(NULL== (fp_log = fopen(logFile,"w"))) {
    printf("%s: error in opening %s\n",argv[0],logFile);
    exit(1);
  }


  /*-------------------------------------
    学習データのファイル数, 各データの
    フレーム数を取得
  -------------------------------------*/
  /* 学習用データのファイル名一覧ファイルを開く */
  if(NULL == (fp_trainingList = fopen(trainingList,"r"))){
    fprintf(stderr,"%s: error in opening %s\n",progname,trainingList);
    exit(1);
  }

  /* 学習用データのファイル数を数える */
  n_token = 0 ;
  while(NULL != fgets(fvFile,FNAMELEN,fp_trainingList)) n_token++;
  rewind(fp_trainingList);

  /* 学習用データ毎のフレーム数を記録する配列 n_frame を生成 */
  if(NULL == (n_frame=(int*)malloc(sizeof(int)*n_token))) {
    fprintf(stderr,"%s: error in allocating n_frame\n",progname);
  }

  /* 学習用データ毎のフレーム数を記録する */
  n_total_frame = 0L;
  i_token = 0;
  while(NULL != fgets(fvFile,FNAMELEN,fp_trainingList)) {
    if('#' != fvFile[0]) {             /* '#'で始まる行は無視する  */
      fvFile[strlen(fvFile)-1] = '\0';
      if(NULL == (fp_fv = fopen(fvFile,"r"))) {
	fprintf(stderr,"%s: error in opening %s\n",argv[0],fvFile);
	exit(1);
      }
      if(1 != fread(&htk,sizeof(HTKHEADER),1,fp_fv)) {
	fprintf(stderr,"%s: error in reading HTK header from %s\n",argv[0],fvFile);
	exit(1);
      }
      n_frame[i_token] = htk.nSamples;   /* フレーム数              */
      n_dim = htk.sampSize/sizeof(float); /* ベクトル次元数…毎回読み
                         込む必要はないが，このように設定しておいて，
                         最後に設定した数値を計算に用いることにする */
      n_total_frame += n_frame[i_token];
      i_token++;
      fclose(fp_fv);
    }
  }
  rewind(fp_trainingList);

  /* 学習データの諸元およびHMMの構成を表示する */
  fprintf(stderr,"training_data= %s\n",trainingList);
  fprintf(stderr,"tokens= %d\n",n_token);
  fprintf(stderr,"frame= %d\n",n_total_frame);
  fprintf(stderr,"dim= %d\n",n_dim);
  fprintf(stderr,"state= %d\n",n_state);
  fprintf(stderr,"mix= %d\n",n_mix);
  fprintf(stderr,"hmm_file= %s\n",hmmFile);
  fprintf(stderr,"log_file= %s\n",logFile);
  fflush(stderr);

  /* 特徴ベクトル時系列を読み込む配列fv( #token * #frame * dim )を生成 */
  if(NULL == (fv=(float***)malloc(sizeof(float**)*n_token))) {
    fprintf(stderr,"%s: error in allocating fv\n",progname);
    exit(1);
  }
  for(i_token=0; i_token<n_token; i_token++) {
    if(NULL == (fv[i_token]=(float**)malloc(sizeof(float*)*n_frame[i_token]))) {
      fprintf(stderr,"%s: error in allocating fv[%d]\n",progname,i_token);
      exit(1); 
    }
    for(i_frame=0; i_frame<n_frame[i_token]; i_frame++) {
      if(NULL == (fv[i_token][i_frame]=(float*)malloc(sizeof(float)*n_dim))) {
	fprintf(stderr,"%s: error in allocating fv[%d][%d]\n",progname,i_token,i_frame);
	exit(1);
      }
    }
  }

  /* 音素学習用トークンの全てのMFCC時系列を配列に読み込む */
  i_token = 0 ;
  while(NULL != fgets(fvFile,FNAMELEN,fp_trainingList)) {
    if('#' != fvFile[0]) {             /* '#'で始まる行は無視する  */
      fvFile[strlen(fvFile)-1] = '\0';
      if(NULL == (fp_fv = fopen(fvFile,"r"))) {
	fprintf(stderr,"%s: error in opening %s\n",progname,fvFile);
	exit(1);
      }
      if(1 != fread(&htk,sizeof(HTKHEADER),1,fp_fv)) {
	fprintf(stderr,"%s: error in reading HTK header from %s\n",progname,fvFile);
	exit(1);
      }
      for(i_frame=0; i_frame<n_frame[i_token]; i_frame++) {
	if(n_dim != fread(fv[i_token][i_frame],sizeof(float),n_dim,fp_fv)) {
	  fprintf(stderr,"%s: error in reading fv[%d] from file %s\n",progname,i_token,fvFile);
	  exit(1);
	}
      }
      i_token ++;
      fclose(fp_fv);
    }
  }
  fclose(fp_trainingList);


  /*-------------------------------------------------------------------
    計算用配列の割り当て
  -------------------------------------------------------------------*/
  /* 前向き確率α ( #token * #frame * #state ) */
  if(NULL == (alpha=(double***)malloc(sizeof(double**)*n_token))) {
    fprintf(stderr,"%s: error in allocating alpha\n",progname);
    exit(1);
  }
  for(i_token=0; i_token<n_token; i_token++) {
    if(NULL == (alpha[i_token]=(double**)malloc(sizeof(double*)*n_frame[i_token]))) {
      fprintf(stderr,"%s: error in allocating alpha[%d]\n",progname,i_token);
      exit(1);
    }
    for(i_frame=0; i_frame<n_frame[i_token]; i_frame++) {
      if(NULL == (alpha[i_token][i_frame]=(double*)malloc(sizeof(double)*n_state))) {
	fprintf(stderr,"%s: error in allocating alpha[%d][%d]\n",progname,i_token,i_frame);
	exit(1);
      }
    }
  }

  /* 後ろ確率β ( #token * #frame * #state ) */
  if(NULL == (beta=(double***)malloc(sizeof(double**)*n_token))) {
    fprintf(stderr,"%s: error in allocating beta\n",progname);
  }
  for(i_token=0; i_token<n_token; i_token++) {
    if(NULL == (beta[i_token]=(double**)malloc(sizeof(double*)*n_frame[i_token]))) {
      fprintf(stderr,"%s: error in allocating beta[%d]\n",progname,i_token);
      exit(1);
    }
    for(i_frame=0; i_frame<n_frame[i_token]; i_frame++) {
      if(NULL == (beta[i_token][i_frame]=(double*)malloc(sizeof(double)*n_state))) {
	fprintf(stderr,"%s: error in allocating beta[%d][%d]\n",progname,i_token,i_frame);
	exit(1);
      }
    }
    
  }

  /* スケーリング係数 scale ( #token * #frame ) */
  if(NULL == (scale=(double**)malloc(sizeof(double*)*n_token))) {
    fprintf(stderr,"%s: error in allocating scale\n",progname);
    exit(1);
  }
  for(i_token=0; i_token<n_token; i_token++) {
    if(NULL == (scale[i_token]=(double*)malloc(sizeof(double)*n_frame[i_token]))) {
      fprintf(stderr,"%s: error in allocating scale[%d]\n",progname,i_token);
      exit(1);
    }
  }

  /* ξ ( #token * #frame * #state * 2 ) */
  if(NULL == (xi=(double****)malloc(sizeof(double***)*n_token))) {
    fprintf(stderr,"%s: error in allocating xi\n",progname);
    exit(1);
  }
  for(i_token=0; i_token<n_token; i_token++) {
    if(NULL == (xi[i_token]=(double***)malloc(sizeof(double**)*n_frame[i_token]))) {
      fprintf(stderr,"%s: error in allocating xi[%d]\n",progname,i_token);
      exit(1);
    }
    for(i_frame=0; i_frame<n_frame[i_token]; i_frame++) {
      if(NULL == (xi[i_token][i_frame]=(double**)malloc(sizeof(double*)*n_state))) {
	fprintf(stderr,"%s: error in allocating xi[%d][%d]\n",progname,i_token,i_frame);
	exit(1);
      }
      for(i_state=0; i_state<n_state; i_state++) {
	if(NULL == (xi[i_token][i_frame][i_state]=(double*)malloc(sizeof(double)*2))) {
	  fprintf(stderr,"%s: error in allocating xi[%d][%d][%d]\n",progname,i_token,i_frame,i_state);
	  exit(1);
	}
      }
    }
  }

  /* γ ( #token * #frame * #state ) */
  if(NULL == (gamma1=(double***)malloc(sizeof(double**)*n_token))) {
    fprintf(stderr,"%s: error in allocating gamma1\n",progname);
    exit(1);
  }
  for(i_token=0; i_token<n_token; i_token++) {
    if(NULL == (gamma1[i_token]=(double**)malloc(sizeof(double*)*n_frame[i_token]))) {
      fprintf(stderr,"%s: error in allocating gamma1[%d]\n",progname,i_token);
      exit(1);
    }
    for(i_frame=0; i_frame<n_frame[i_token]; i_frame++) {
      if(NULL == (gamma1[i_token][i_frame]=(double*)malloc(sizeof(double)*n_state))) {
	fprintf(stderr,"%s: error in allocating gamma1[%d][%d]\n",progname,i_token,i_frame);
	exit(1);
      }
    }
    
  }

  /* ζ　( #token * #frame * #state * n_mix ) */
  if(NULL == (zeta=(double****)malloc(sizeof(double***)*n_token))) {
    fprintf(stderr,"%s: error in allocating zeta\n",progname);
  }
  for(i_token=0; i_token<n_token; i_token++) {
    if(NULL == (zeta[i_token]=(double***)malloc(sizeof(double**)*n_frame[i_token]))) {
      fprintf(stderr,"%s: error in allocating zeta[%d]\n",progname,i_token);
      exit(1);
    }
    for(i_frame=0; i_frame<n_frame[i_token]; i_frame++) {
      if(NULL == (zeta[i_token][i_frame]=(double**)malloc(sizeof(double*)*n_state))) {
	fprintf(stderr,"%s: error in allocating zeta[%d][%d]\n",progname,i_token,i_frame);
	exit(1);
      }
      for(i_state=0; i_state<n_state; i_state++) {
	if(NULL == (zeta[i_token][i_frame][i_state]=(double*)malloc(sizeof(double)*n_mix))) {
	  fprintf(stderr,"%s: error in allocating zeta[%d][%d][%d]\n",progname,i_token,i_frame,i_state);
	  exit(1);
	}
      }
    }
  }

  /*------------------------------------
    HMMの学習
   ------------------------------------*/
  /* HMMパラメータ配列の生成 */
  if(NULL == (hmm=(HMM*)malloc(sizeof(HMM)*n_state))){
    fprintf(stderr,"%s: error in allocating hmm\n",progname);
    exit(1);
  }
  for(i_state=0; i_state<n_state; i_state++) {
    if(NULL == (hmm[i_state].a=(float*)malloc(sizeof(float)*2))){
      fprintf(stderr,"%s: error in allocating hmm.a\n",progname);
      exit(1);
    }
    if(NULL == (hmm[i_state].c=(float*)malloc(sizeof(float)*n_mix))){
      fprintf(stderr,"%s: error in allocating hmm.c\n",progname);
      exit(1);
    }
    if(NULL == (hmm[i_state].mu=(float**)malloc(sizeof(float*)*n_mix))){
      fprintf(stderr,"%s: error in allocating hmm.mu\n",progname);
      exit(1);
    }
    for(i_mix=0; i_mix<n_mix; i_mix++) {
      if(NULL == (hmm[i_state].mu[i_mix]=(float*)malloc(sizeof(float)*n_dim))){
	fprintf(stderr,"%s: error in allocating hmm.mu[%d]\n",progname,i_mix);
	exit(1);
      }
    }
    if(NULL == (hmm[i_state].sigma2=(float**)malloc(sizeof(float*)*n_mix))){
      fprintf(stderr,"%s: error in allocating hmm.sigma2\n",progname);
      exit(1);
    }
    for(i_mix=0; i_mix<n_mix; i_mix++) {
      if(NULL == (hmm[i_state].sigma2[i_mix]=(float*)malloc(sizeof(float)*n_dim))){
	fprintf(stderr,"%s: error in allocating hmm.sigma2[%d]\n",progname,i_mix);
	exit(1);
      }
    }
  }
  if(NULL == (hmm_old=(HMM*)malloc(sizeof(HMM)*n_state))){
    fprintf(stderr,"%s: error in allocating hmm_old\n",progname);
    exit(1);
  }
  for(i_state=0; i_state<n_state; i_state++) {
    if(NULL == (hmm_old[i_state].a=(float*)malloc(sizeof(float)*2))){
      fprintf(stderr,"%s: error in allocating hmm_old.a\n",progname);
      exit(1);
    }
    if(NULL == (hmm_old[i_state].c=(float*)malloc(sizeof(float)*n_mix))){
      fprintf(stderr,"%s: error in allocating hmm_old.c\n",progname);
      exit(1);
    }
    if(NULL == (hmm_old[i_state].mu=(float**)malloc(sizeof(float*)*n_mix))){
      fprintf(stderr,"%s: error in allocating hmm_old.mu\n",progname);
      exit(1);
    }
    for(i_mix=0; i_mix<n_mix; i_mix++) {
      if(NULL == (hmm_old[i_state].mu[i_mix]=(float*)malloc(sizeof(float)*n_dim))){
	fprintf(stderr,"%s: error in allocating hmm_old.mu[%d]\n",progname,i_mix);
	exit(1);
      }
    }
    if(NULL == (hmm_old[i_state].sigma2=(float**)malloc(sizeof(float*)*n_mix))){
      fprintf(stderr,"%s: error in allocating hmm_old.sigma2\n",progname);
      exit(1);
    }
    for(i_mix=0; i_mix<n_mix; i_mix++) {
      if(NULL == (hmm_old[i_state].sigma2[i_mix]=(float*)malloc(sizeof(float)*n_dim))){
	fprintf(stderr,"%s: error in allocating hmm_old.sigma2[%d]\n",progname,i_mix);
	exit(1);
      }
    }
  }

  /* HMMパラメータの初期化 */
  parm_init(n_mix,n_dim,n_token,n_frame,n_total_frame,fv,n_state,hmm);

  /* 再推定の繰り返し */
  reest_count = 0;
  logp_new = re_estimate1(n_mix,n_dim,n_token,n_frame,fv,n_state,hmm,alpha,beta,scale,xi,gamma1,zeta);
  fprintf(stderr,"(%d) logp= %e\n",reest_count,logp_new);
  fprintf(fp_log,"(%d) logp= %e\n",reest_count,logp_new);
  for(i_state=0; i_state<n_state; i_state++){
    hmm_old[i_state] = hmm[i_state] ;
  }
  best_num = 0 ;
  best_min = INFNTY ;
  do {
    reest_count++;
    logp_old = logp_new;
    logp_new = re_estimate1(n_mix,n_dim,n_token,n_frame,fv,n_state,hmm,
			    alpha,beta,scale,xi,gamma1,zeta);
    fprintf(stderr,"(%d) logp= %e\n",reest_count,logp_new); fflush(stderr);
    fprintf(fp_log,"(%d) logp= %e\n",reest_count,logp_new); fflush(fp_log);
    if(!nan_inf_determin(logp_new)){
      if(best_min > (logp_new - logp_old)){
	best_min = logp_new - logp_old;
	best_num = reest_count;
	for(i_state=0; i_state<n_state; i_state++){
	  hmm_old[i_state] = hmm[i_state] ;
	}
      }
    }

    if(0 != nan_inf_determin(logp_new)) {
      fprintf(stderr,"hmm_old copy to hmm\n");
      fprintf(stderr,"best_num = %d , best_min = %e\n" , best_num , best_min);
      fclose(fp_log);
      for(i_state=0; i_state<n_state; i_state++){
	hmm[i_state] = hmm_old[i_state];
      }
      break ;
    }
  } while((logp_new-logp_old)>EPSILON);

  /*-------------------------------------------
    学習したHMMパラメータをファイルに書き出す
  -------------------------------------------*/
  if(NULL == (fp_hmm = fopen(hmmFile,"w"))) {
    fprintf(stderr,"%s: error in opening %s\n",progname,hmmFile);
    exit(1);
  }

  /* 特徴ベクトル次元,混合数,状態数 */
  fprintf(fp_hmm,"dim= %d\n",n_dim);
  fprintf(fp_hmm,"state= %d\n",n_state);
  fprintf(fp_hmm,"mix= %d\n",n_mix);

  /* 状態遷移確率 */
  for(i_state=0; i_state<n_state; i_state++) {
    for(j_state=0; j_state<2; j_state++){
      fprintf(fp_hmm,"%e ",hmm[i_state].a[j_state]);
    }
    fprintf(fp_hmm,"\n");
    /* 混合Gaussianパラメータ */
    for(i_mix=0; i_mix<n_mix; i_mix++) {
      /* 混合係数 */
      fprintf(fp_hmm,"%e\n",hmm[i_state].c[i_mix]);
      /* 平均 */
      for(i_dim=0; i_dim<n_dim; i_dim++){
	fprintf(fp_hmm,"%e ",hmm[i_state].mu[i_mix][i_dim]);
      }
      fprintf(fp_hmm,"\n");
      /* 分散 */
      for(i_dim=0; i_dim<n_dim; i_dim++){
	fprintf(fp_hmm,"%e ",hmm[i_state].sigma2[i_mix][i_dim]);
      }
      fprintf(fp_hmm,"\n");
    }
  }
  fclose(fp_hmm);
  fclose(fp_log);


  exit(EXIT_SUCCESS);
}




/*---------------------------------------------------------------------

  parm_init

---------------------------------------------------------------------*/
void parm_init(int n_mix,                /* 混合数                   */
	       int n_dim,                /* 特徴ベクトル次元数       */
	       int n_token,              /* 学習データ数             */
	       int *n_frame,             /* 各トークンのフレーム数   */
	       int n_total_frame,        /* 学習データのフレーム総数 */
	       float ***fv,              /* 特徴ベクトル             */
	       int n_state,              /* HMM状態数                */
	       HMM *hmm)                 /* HMMパラメータ            */
{
  int **start;                           /* (n_state*n_token)        */
  int **end;                             /* (n_state*n_token)        */
  int i_token,i_state,j_state,i_frame,i_mix,i_dim;
  int quot, rem;                         /* quotient, remainder      */
  int st, et;                            /* temporary start and end  */
  float **centroid;                      /* (n_mix*n_dim)            */
  int **tokentable;                      /* (n_mix*n_total_frame)    */
  int **frametable; /* frame # belonging to a cluster (n_mix*n_total_frame)*/
  int *numframe;          /* #frame belonging to a component (n_mix) */
  float temp;
  double sigmatemp;                      /* temporary buffer         */

  

  /*-------------------------------------
    計算用の配列を割り当てる
  -------------------------------------*/
  /* 各状態の分布の初期値計算に用いるデータの開始・終了フレーム */
  if(NULL == (start=(int**)malloc(sizeof(int*)*n_state))) {
    fprintf(stderr,"%s: error in allocating start\n",progname);
    exit(1);
  }
  for(i_state=0; i_state<n_state; i_state++) {
    if(NULL == (start[i_state]=(int*)malloc(sizeof(int)*n_token))) {
      fprintf(stderr,"%s: error in allocating start[%d]\n",progname,i_state);
      exit(1);
    }
  }
  if(NULL == (end=(int**)malloc(sizeof(int*)*n_state))) {
    fprintf(stderr,"%s: error in allocating end\n",progname);
    exit(1);
  }
  for(i_state=0; i_state<n_state; i_state++) {
    if(NULL == (end[i_state]=(int*)malloc(sizeof(int)*n_token))) {
      fprintf(stderr,"%s: error in allocating end[%d]\n",progname,i_state);
      exit(1);
    }
  }
  /* クラスターのセントロイド */
  if(NULL == (centroid=(float**)malloc(sizeof(float*)*n_mix))) {
    fprintf(stderr,"%s: error in allocating centroid\n",progname);
    exit(1);
  }
  for(i_mix=0; i_mix<n_mix; i_mix++) {
    if(NULL == (centroid[i_mix]=(float*)malloc(sizeof(float)*n_dim))) {
      fprintf(stderr,"%s: error in allocating centroid[%d]\n",progname,i_state);
      exit(1);
    }
  }
  /* クラスターのメンバーに関する配列 */
  if(NULL == (tokentable=(int**)malloc(sizeof(int*)*n_mix))) {
    fprintf(stderr,"%s: error in allocating tokentable\n",progname);
    exit(1);
  }
  for(i_mix=0; i_mix<n_mix; i_mix++) {
    if(NULL == (tokentable[i_mix]=(int*)malloc(sizeof(int)*n_total_frame))) {
      fprintf(stderr,"%s: error in allocating tokentable[%d]\n",progname,i_mix);
      exit(1);
    }
  }
  if(NULL == (frametable=(int**)malloc(sizeof(int*)*n_mix))) {
    fprintf(stderr,"%s: error in allocating frametable\n",progname);
    exit(1);
  }
  for(i_mix=0; i_mix<n_mix; i_mix++) {
    if(NULL == (frametable[i_mix]=(int*)malloc(sizeof(int)*n_total_frame))) {
      fprintf(stderr,"%s: error in allocating frametable[%d]\n",progname,i_mix);
      exit(1);
    }
  }
  /* 各クラスタのメンバー(フレーム)数 */
  if(NULL == (numframe=(int*)malloc(sizeof(int)*n_mix))) {
    fprintf(stderr,"%s: error in allocating numframe\n",progname);
    exit(1);
  }

  /*-----------------------------------------------
    各学習データの特徴時系列を状態数で等分割する
  -----------------------------------------------*/
  for(i_token=0; i_token<n_token; i_token++) {
    quot = n_frame[i_token]/n_state;
    rem = n_frame[i_token]%n_state;
    et = -1;
    for(i_state=0; i_state<rem; i_state++) {
      st = et+1;
      et = et+quot+1;
      start[i_state][i_token] = st;
      end[i_state][i_token] = et;
    }
    for(;i_state<n_state; i_state++) {
      st = et+1;
      et = et+quot;
      start[i_state][i_token] = st;
      end[i_state][i_token] = et;
    }
  }
  
  /* LBG clustering for each state */
  for(i_state=0; i_state<n_state; i_state++) {
    n_total_frame = lbg(n_mix,n_dim,n_token,fv,start[i_state],end[i_state],tokentable,frametable,numframe,centroid);

    /* estimate parameter values */
    for(i_mix=0; i_mix<n_mix; i_mix++) {
      for(i_dim=0; i_dim<n_dim; i_dim++) {
	hmm[i_state].mu[i_mix][i_dim] = centroid[i_mix][i_dim]; /* initial estimate of mu */
	sigmatemp = 0.0;
	for(i_frame=0; i_frame<numframe[i_mix]; i_frame++) {
	  temp	= fv[tokentable[i_mix][i_frame]][frametable[i_mix][i_frame]][i_dim]-centroid[i_mix][i_dim];
	  if(temp == 0.0){
	    fprintf(fp_log,"ut=%d ft=%d\n",tokentable[i_mix][i_frame],frametable[i_mix][i_frame]);
	  }
	  sigmatemp += temp*temp;
	}
	hmm[i_state].sigma2[i_mix][i_dim] = sigmatemp/numframe[i_mix];
      }
    }
    for(i_mix=0; i_mix<n_mix; i_mix++) {
      hmm[i_state].c[i_mix] = (float)numframe[i_mix]/(float)n_total_frame;
    }			  /* initial estimate of mixture weight */
  }

  /* initial value of a */
  for(i_state=0; i_state<n_state; i_state++) {
    for(j_state=0; j_state<2; j_state++) {
      hmm[i_state].a[j_state] = 0.5;
    }
  }

  /*-------------------------------------
    計算用の配列の開放
  -------------------------------------*/
  for(i_state=0; i_state<n_state; i_state++) free(start[i_state]);
  free(start);
  for(i_state=0; i_state<n_state; i_state++) free(end[i_state]);
  free(end);
  for(i_mix=0; i_mix<n_mix; i_mix++) free(centroid[i_mix]);
  free(centroid);
  for(i_mix=0; i_mix<n_mix; i_mix++) free(tokentable[i_mix]);
  free(tokentable);
  for(i_mix=0; i_mix<n_mix; i_mix++) free(frametable[i_mix]);
  free(frametable);
  free(numframe);

}



/*---------------------------------------------------------------------

  lbg

---------------------------------------------------------------------*/
int lbg(int n_mix,                       /* 混合数                   */
	int n_dim,                       /* 特徴ベクトル次元数       */
	int n_token,                     /* 学習データ数             */
	float ***fv,                     /* 特徴ベクトル             */
	int *start,
	int *end,
	int **tokentable,
	int **frametable,
	int *numframe,
	float **centroid)
{
  int i;
  int n_stage;            /* number of clustering stages */
  int i_stage;            /* clustering stage counter */
  int n_cluster;
  int i_token,i_frame,i_mix,i_dim;
  int n_total_frame;
  float delta;           /* perturbation factor */


  /* セントロイドの初期化 */
  n_cluster = 1;             /* クラスタ数の初期値 */
  i_mix = 0;
  for(i_dim=0; i_dim<n_dim; i_dim++) {
    centroid[i_mix][i_dim] = 0.0;   /* all clear */
  }
  n_total_frame = 0;
  for(i_token = 0; i_token < n_token; i_token++) {
    for(i_frame=start[i_token]; i_frame<=end[i_token]; i_frame++) {
      n_total_frame++;
      for(i_dim=0; i_dim<n_dim; i_dim++) {
	centroid[i_mix][i_dim] += fv[i_token][i_frame][i_dim]; /*sum of frame data*/
      }
    }
  }
  for(i_dim=0; i_dim<n_dim; i_dim++) {
    centroid[i_mix][i_dim] /= n_total_frame; /* averaging */
  }
  
  n_stage = ilog2(n_mix);
  if(n_stage < 0) {
    fprintf(stderr , "n_mix must be a power of 2\n");
    exit(1);
  }

  /* 所望のクラスタ数になるまで,セントロイドを分割し
     k-meansクラスタリングを行う                     */
  for(i_stage=1; i_stage<=n_stage; i_stage++) {
    /* セントロイドの分割 */
    for(i_mix=0; i_mix<n_cluster; i_mix++) {
      for(i_dim=0; i_dim<n_dim; i_dim++){
	centroid[i_mix+n_cluster][i_dim] = centroid[i_mix][i_dim]; /* copy */
      }
    }
    for(i_mix=0; i_mix<n_cluster; i_mix++) {
      for(i_dim=0; i_dim<n_dim; i_dim++) {
	delta = (float)(DELTA*(rand()/2.147e+09-0.5));
	centroid[i_mix][i_dim] += delta;         /* split */
	centroid[i_mix+n_cluster][i_dim] -= delta;  /* split */
      }
    }
    n_cluster *= 2;  /* double the number of clusters */
    kmeans(n_mix,n_dim,n_token,fv,start,end,n_total_frame,n_cluster,tokentable,frametable,numframe,centroid);
  }

  return (n_total_frame);
}



/*----------------------------------------------------------------------

 kmeans

----------------------------------------------------------------------*/
int kmeans(int n_mix,
           int n_dim,
	   int n_token,
	   float ***fv,
	   int *start,
	   int *end,
	   int n_total_frame,
	   int n_cluster,
	   int **tokentable,
	   int **frametable,
	   int *numframe,
	   float **centroid)
{
  int i_token,i_frame, i_cluster, i_dim;
  float *distort;                        /*クラスターの歪み          */
  float distort_old, distort_new;        /* 総歪み                   */
  float distance;      /* フレームとセントロイド間のユークリッド距離 */
  float mindist;                         /* 距離の最小値             */
  int minclust;            /* cluster number giving minimum distance */
  
  

  /*-------------------------------------
    計算用の配列を割り当てる
  -------------------------------------*/
  /* クラスターの歪み */
  if(NULL == (distort=(float*)malloc(sizeof(float)*n_mix))) {
    fprintf(stderr,"%s: error in allocating distort\n",progname);
    exit(1);
  }

  /*-------------------------------------
    クラスタリング
  -------------------------------------*/
  distort_new = INFNTY;
  do {
    for(i_cluster=0; i_cluster<n_cluster; i_cluster++) {
	numframe[i_cluster] = 0;
    }
    distort_old = distort_new; /* save distort_new */
      
    /* 各フレームの特徴ベクトルの最近傍のセントロイドを探す */
    for(i_token=0; i_token<n_token; i_token++) {
      for(i_frame=start[i_token]; i_frame<=end[i_token]; i_frame++) {
	minclust = 0;
	mindist = INFNTY;
	for(i_cluster=0; i_cluster<n_cluster; i_cluster++) {
	  distance = euclid(centroid[i_cluster],fv[i_token][i_frame],n_dim);
	  if(distance<mindist) {
	    mindist = distance;
	    minclust = i_cluster;
	  }
	}
	tokentable[minclust][numframe[minclust]] = i_token;
	frametable[minclust][numframe[minclust]] = i_frame;
	numframe[minclust] ++;
      }
    }

    /* セントロイドの更新 */
    for(i_cluster=0; i_cluster<n_cluster; i_cluster++) {
      for(i_dim=0; i_dim<n_dim; i_dim++) {
	centroid[i_cluster][i_dim] = 0.0 ;
	for(i_frame=0; i_frame<numframe[i_cluster]; i_frame++) {
	  centroid[i_cluster][i_dim]
	    += fv[tokentable[i_cluster][i_frame]][frametable[i_cluster][i_frame]][i_dim];
	}
      }
    }
    for(i_cluster=0; i_cluster<n_cluster; i_cluster++) {
      for (i_dim=0; i_dim<n_dim; i_dim++){
	centroid[i_cluster][i_dim] /= numframe[i_cluster];
      }
    }

    /* new distortion */
    for(i_cluster=0; i_cluster<n_cluster; i_cluster++) {
      distort[i_cluster] = 0.0;
      for(i_frame=0; i_frame<numframe[i_cluster]; i_frame++) {
	distance = euclid(centroid[i_cluster],
			  fv[tokentable[i_cluster][i_frame]][frametable[i_cluster][i_frame]],
			  n_dim);
	distort[i_cluster] += distance;
      }
    }
    
    distort_new = 0.0;
    for(i_cluster=0; i_cluster<n_cluster; i_cluster++){
      distort_new += distort[i_cluster];
    }
    distort_new /= n_total_frame;

  } while (LAMBDA < (distort_old-distort_new)/distort_old);


  /* 計算用の配列の開放 */
  free(distort);

  return 0;
}





/*----------------------------------------------------------------------

 euclid distance

----------------------------------------------------------------------*/
float euclid(float *v1,
	     float *v2,
	     int n_dim)
{
  int i_dim;
  float dist, result;

  
  result = 0.0;
  for(i_dim=0; i_dim<n_dim; i_dim++) {
    dist = v1[i_dim]-v2[i_dim];
    result += dist*dist;
  }
  return (result);

}




/*----------------------------------------------------------------------

 re_estimate1

----------------------------------------------------------------------*/
float re_estimate1(int n_mix,
		   int n_dim,
		   int n_token,
		   int *n_frame,
		   float ***fv,
		   int n_state,
		   HMM *hmm,
		   double ***alpha,
		   double ***beta,
		   double **scale,
		   double ****xi,
		   double ***gamma1,
		   double ****zeta)
{
  int i_token,i_frame,i_state,j_state,i_mix,i_dim;
  int s;                                 /* 状態数のバッファ         */
  float *logp;                     /* log probability for each token */
  float logprob;                   /* log probability for all tokens */
  double *gamma1sum;                     /* γの累算値               */
  double sum1, sum2, sum3;               /* accumulater              */
  double diff;                           /* temporary buffer         */
  

  /*-------------------------------------
    計算用の配列を割り当てる
  -------------------------------------*/
  /* 各トークンに対する対数尤度 */
  if(NULL == (logp=(float*)malloc(sizeof(float)*n_token))) {
    fprintf(stderr,"%s: error in allocating logp\n",progname);
    exit(1);
  }   
  /* γの累算値 */
  if(NULL == (gamma1sum=(double*)malloc(sizeof(double)*n_state))) {
    fprintf(stderr,"%s: error in allocating gamma1sum\n",progname);
    exit(1);
  }   


  /*-------------------------------------
    Baum-Welch
  -------------------------------------*/
  for(i_token=0; i_token<n_token; i_token++) {
    forward(n_dim,i_token,n_frame,fv,n_mix,alpha,scale,n_state,hmm);
  }
  for(i_token=0; i_token<n_token; i_token++){
    backward(n_dim,i_token,n_frame,fv,n_mix,beta,scale,n_state,hmm);
  }

  logprob = 0.0;
  for(i_token=0; i_token<n_token; i_token++) {
    logp[i_token] = 0.0;
    for(i_frame=0; i_frame<n_frame[i_token]; i_frame++) {
      logp[i_token] += -(float)log(scale[i_token][i_frame]);
    }
    logp[i_token] += (float)log(alpha[i_token][n_frame[i_token]-1][n_state-1])+(float)log(hmm[n_state-1].a[1]);
    logprob += logp[i_token];
  }
  logprob /= n_token;                        /* トークン数で正規化 */

  for(i_token=0; i_token<n_token; i_token++) {
    for(i_frame=0; i_frame<n_frame[i_token]-2; i_frame++) {
      for(i_state=0; i_state<n_state-1; i_state++) {
	for(j_state=0; j_state<2; j_state++) {
	  s = i_state+j_state;
	  xi[i_token][i_frame][i_state][j_state]
	    = alpha[i_token][i_frame][i_state]
	      *beta[i_token][i_frame+1][s]
		*hmm[i_state].a[j_state]
		  *gauss_mix(n_dim,n_mix,i_token,i_frame+1,fv,hmm[s].mu,hmm[s].sigma2,hmm[s].c)
		    /(alpha[i_token][n_frame[i_token]-1][n_state-1]*hmm[n_state-1].a[1]);
	}
      }
      i_state = n_state-1;
      j_state = 0;
      s = i_state+j_state ;
      xi[i_token][i_frame][i_state][j_state]
	= alpha[i_token][i_frame][i_state]
	    *beta[i_token][i_frame+1][s]
	      *hmm[i_state].a[j_state]
	        *gauss_mix(n_dim,n_mix,i_token,i_frame+1,fv,hmm[s].mu,hmm[s].sigma2,hmm[s].c)
                  /(alpha[i_token][n_frame[i_token]-1][n_state-1]
                    *hmm[n_state-1].a[1]);
      j_state = 1;
      xi[i_token][i_frame][i_state][j_state] = 0.0;
    }

    i_frame = n_frame[i_token]- 2;
    for(i_state=0; i_state<n_state-2; i_state++) {
      for(j_state=0; j_state<2; j_state++) {
	xi[i_token][i_frame][i_state][j_state] = 0.0;
      }
    }
    i_state = n_state-2;
    j_state = 0;
    xi[i_token][i_frame][i_state][j_state] = 0.0;
    j_state = 1;

    xi[i_token][i_frame][i_state][j_state]
      = alpha[i_token][i_frame][i_state]*beta[i_token][i_frame][i_state]
	  /(scale[i_token][i_frame]
	    *alpha[i_token][n_frame[i_token]-1][n_state-1]
	      *hmm[n_state-1].a[1]);

    i_state = n_state-1;
    j_state = 0;
    xi[i_token][i_frame][i_state][j_state]
      = alpha[i_token][i_frame][i_state]*beta[i_token][i_frame][i_state]
          /(scale[i_token][i_frame]
	   *alpha[i_token][n_frame[i_token]-1][n_state-1]
             *hmm[n_state-1].a[1]);
    j_state = 1;
    xi[i_token][i_frame][i_state][j_state] = 0.0;
    i_frame = n_frame[i_token]-1;
    
    for(i_state=0; i_state<n_state-1; i_state++) {
      for(j_state=0; j_state<2; j_state++) {
	xi[i_token][i_frame][i_state][j_state] = 0.0;
      }
    }
    
    i_state = n_state-1;
    j_state = 0;
    xi[i_token][i_frame][i_state][j_state] = 0.0;
    j_state = 1;
    xi[i_token][i_frame][i_state][j_state] = 1.0;
  }

  for(i_token=0; i_token<n_token; i_token++) {
    for(i_frame=0; i_frame<n_frame[i_token]-1; i_frame++) {
      for(i_state=0; i_state<n_state; i_state++) {
	gamma1[i_token][i_frame][i_state]
	  = alpha[i_token][i_frame][i_state]
	      *beta[i_token][i_frame][i_state]
	        /(scale[i_token][i_frame]
		  *alpha[i_token][n_frame[i_token]-1][n_state-1]
		    *hmm[n_state-1].a[1]);
      }
    }
    i_frame = n_frame[i_token]-1;
    for(i_state=0; i_state<n_state-1; i_state++){
      gamma1[i_token][i_frame][i_state] = 0.0;
    }
    i_state = n_state-1;
    gamma1[i_token][i_frame][i_state] = 1.0;
  }
  
  for(i_token=0; i_token<n_token; i_token++) {
    for(i_frame=0; i_frame<n_frame[i_token]; i_frame++) {
      for(i_state=0; i_state<n_state; i_state++) {
	for(i_mix=0; i_mix<n_mix; i_mix++) {
	  zeta[i_token][i_frame][i_state][i_mix]
	    = gamma1[i_token][i_frame][i_state]
	        *hmm[i_state].c[i_mix]
		  *gauss(n_dim,i_frame,fv[i_token],hmm[i_state].mu[i_mix],hmm[i_state].sigma2[i_mix])
		    /gauss_mix(n_dim,n_mix,i_token,i_frame,fv,hmm[i_state].mu,
			       hmm[i_state].sigma2,hmm[i_state].c);
	}
      }
    }
  }

  for(i_state=0; i_state<n_state; i_state++) {
    gamma1sum[i_state] = 0.0;
    for(i_token=0; i_token<n_token; i_token++) {
      for(i_frame=0; i_frame<n_frame[i_token]; i_frame++) {
	gamma1sum[i_state] += gamma1[i_token][i_frame][i_state];
      }
    }
  }

  for(i_state=0; i_state<n_state; i_state++) {
    for(j_state=0; j_state<2; j_state++) {
      sum1 = 0.0;
      for(i_token=0; i_token<n_token; i_token++) {
	for (i_frame=0; i_frame<n_frame[i_token]; i_frame++){
	  sum1 += xi[i_token][i_frame][i_state][j_state];
	}
      }
      hmm[i_state].a[j_state] = (float) (sum1 / gamma1sum[i_state]);
    }
  }

  for(i_state=0; i_state<n_state; i_state++) {
    for(i_mix=0; i_mix<n_mix; i_mix++) {
      sum1 = 0.0;
      for(i_token=0; i_token<n_token; i_token++){
	for(i_frame=0; i_frame<n_frame[i_token]; i_frame++){
	  sum1 += zeta[i_token][i_frame][i_state][i_mix];
	}
      }
      hmm[i_state].c[i_mix] = (float)(sum1/gamma1sum[i_state]);

      for(i_dim=0; i_dim<n_dim; i_dim++) {
	sum2 = 0.0;
	for(i_token=0; i_token<n_token; i_token++) {
	  for(i_frame=0; i_frame<n_frame[i_token]; i_frame++){
	    sum2 += zeta[i_token][i_frame][i_state][i_mix]*fv[i_token][i_frame][i_dim];
	  }
	}
	hmm[i_state].mu[i_mix][i_dim] = (float)(sum2/sum1);
	
	sum3 = 0.0;
	for(i_token=0; i_token<n_token; i_token++) {
	  for (i_frame=0; i_frame<n_frame[i_token]; i_frame++) {
	    diff = fv[i_token][i_frame][i_dim]-hmm[i_state].mu[i_mix][i_dim];
	    sum3 += zeta[i_token][i_frame][i_state][i_mix]*diff*diff;
	  }
	}
	hmm[i_state].sigma2[i_mix][i_dim] = (float)(sum3/sum1);
      }
    }
  }
  return (logprob);
}



/*----------------------------------------------------------------------

 forward

----------------------------------------------------------------------*/
int forward(int n_dim,
	    int i_token,
	    int *n_frame,
	    float ***fv,
	    int n_mix,
	    double ***alpha,
	    double **scale,
	    int n_state,
	    HMM *hmm)
{
  int i,j,k;
  int i_frame, i_state;
  double sum;
  i_frame = 0;
  i_state = 0;


  alpha[i_token][i_frame][i_state]
    = gauss_mix(n_dim,n_mix,i_token,i_frame,fv,
		hmm[i_state].mu,hmm[i_state].sigma2,hmm[i_state].c);
  scale[i_token][i_frame] = 1.0/alpha[i_token][i_frame][i_state];
  alpha[i_token][i_frame][i_state] = 1.0;

  for(i_state=1; i_state<n_state; i_state++) {
    alpha[i_token][i_frame][i_state] = 0.0;
  }

  for(i_frame=1; i_frame<n_frame[i_token]; i_frame++) {
    i_state = 0;
    alpha[i_token][i_frame][i_state]
      = alpha[i_token][i_frame-1][i_state]*hmm[i_state].a[0]
	* gauss_mix(n_dim,n_mix,i_token,i_frame,fv,hmm[i_state].mu,
		    hmm[i_state].sigma2,hmm[i_state].c);
    
    for(i_state=1; i_state<n_state; i_state++) {
      alpha[i_token][i_frame][i_state]
	= (alpha[i_token][i_frame-1][i_state-1]*hmm[i_state-1].a[1]
	   +alpha[i_token][i_frame-1][i_state]*hmm[i_state].a[0])
	   *gauss_mix(n_dim,n_mix,i_token,i_frame,fv,hmm[i_state].mu,
		      hmm[i_state].sigma2,hmm[i_state].c);
    }
    sum = 0.0;
    for(i_state=0; i_state<n_state; i_state++) {
      sum += alpha[i_token][i_frame][i_state];
    }
    
    scale[i_token][i_frame] = 1.0/sum;
    
    for(i_state=0; i_state<n_state; i_state++) {
      alpha[i_token][i_frame][i_state] *= scale[i_token][i_frame];
    }
  }
  
  return (0);
}



/*----------------------------------------------------------------------

 backward

----------------------------------------------------------------------*/
int backward(int n_dim,
	     int i_token,
	     int *n_frame,
	     float ***fv,
	     int n_mix,
	     double ***beta,
	     double **scale,
	     int n_state,
	     HMM *hmm)
{
  int i_frame, i_state;
  

  i_frame = n_frame[i_token]-1;
  i_state = n_state-1;
  beta[i_token][i_frame][i_state] = scale[i_token][i_frame] * hmm[i_state].a[1];
  for(i_state=n_state-2; i_state>=0; i_state--) {
    beta[i_token][i_frame][i_state] = 0.0;
  }
  for(i_frame=n_frame[i_token]-2; i_frame>=0; i_frame--) {
    i_state = n_state-1;
    beta[i_token][i_frame][i_state]
      = hmm[i_state].a[0]
	*gauss_mix(n_dim,n_mix,i_token,i_frame+1,fv,
		    hmm[i_state].mu,hmm[i_state].sigma2, hmm[i_state].c)
	  *beta[i_token][i_frame+1][i_state];
    
    for(i_state=n_state-2; i_state>= 0; i_state--) {
      beta[i_token][i_frame][i_state]
	= hmm[i_state].a[0]
	  *gauss_mix(n_dim,n_mix,i_token,i_frame+1,fv,
		      hmm[i_state].mu,hmm[i_state].sigma2,hmm[i_state].c)
	    *beta[i_token][i_frame+1][i_state]
	      +hmm[i_state].a[1]
		*gauss_mix(n_dim,n_mix,i_token,i_frame+1,fv,hmm[i_state+1].mu,
			   hmm[i_state+1].sigma2,hmm[i_state+1].c)
		  *beta[i_token][i_frame+1][i_state+1];
    }
    for(i_state=n_state-1; i_state>=0; i_state--) {
      beta[i_token][i_frame][i_state] *= scale[i_token][i_frame];
    }
  }
	
  return (0);
}



/*----------------------------------------------------------------------

 gauss_mix

----------------------------------------------------------------------*/
double gauss_mix(int n_dim,
		 int n_mix,
		 int i_token,
		 int i_frame,
		 float ***fv,
		 float **mu,
		 float **sigma2,
		 float *c)
{
  int i_mix;
  double result;


  result = 0.0;
  for(i_mix=0; i_mix<n_mix; i_mix++) {
    result += (double)c[i_mix]*gauss(n_dim,i_frame,fv[i_token],mu[i_mix],sigma2[i_mix]);
  }

  if(result<MINDOUBLE) {
    result = MINDOUBLE;
  }
  
  return (result);
}



/*----------------------------------------------------------------------

 gauss

----------------------------------------------------------------------*/
double gauss(int n_dim,
	     int i_frame,
	     float **fv,
	     float *mu,
	     float *sigma2)
{
  return gpdf(n_dim,fv[i_frame],mu,sigma2);
}



/*----------------------------------------------------------------------

 ilog2

----------------------------------------------------------------------*/
int ilog2(int n)  /* log2 of interger n */
{
  int quat,quat1,rem; /* quatient and remainder */
  int counter;        /* counter */
  
  if(n <= 0) {
    return (-1); /* error */
  }

  /* initialize */
  quat1 = n;
  counter = 0;

  while(quat1 > 1) {
    quat = quat1 / 2;   /* quatient */
    rem = quat1 % 2;    /* remainder */
    
    if(rem != 0) {
      return (-2); /* n is not a power of 2 */
    }
    else {
      quat1 = quat;
      counter++;
    }
  }
  return (counter);
}



/*----------------------------------------------------------------------

 nan_inf_determin
        NaNかInfか-Infかを判定する。それぞれ1,2,3(全て真)が返される．
        一般的な値ならば、0(偽)が返される。

----------------------------------------------------------------------*/
int nan_inf_determin(float f) {
  char	nan_inf[64] ;


  sprintf(nan_inf ,"%e",f);

  if(0 == strcmp(nan_inf,"NaN")) {
    return 1;
  }
  else if(0 == strcmp(nan_inf,"Inf")) {
    return 2 ;
  }
  else if(0 == strcmp(nan_inf,"-Inf")) {
    return 3;
  }
  
  return 0;
}
