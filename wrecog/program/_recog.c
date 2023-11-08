/**********************************************************************

 単語音声認識プログラム(Viterbiデコーダー)〜ストリーム入出力版〜

  使用法: source | recog HMMList
              source: HTK形式MFCCファイルをstdoutに出力し続けるコマンド
             HMMlist: 単語HMMパラメータファイル一覧


   単語HMMで認識するように変更                 2013年09月09日 高木一幸
   (HMMの状態数,Gaussian混合数,特徴パラメータ
    次元数はHMMパラメータファイルから読んで対応する)
   単語番号はHMMファイル名中の文字列 "WD???" として読み取り,
   認識結果に表示する                          2013年09月19日 高木一幸
   名称を "recogW" から "recogf" に変更        2013年10月07日 高木一幸
   関数vtbをviterbiに改名し別ファイルにする    2013年10月07日 高木一幸
   HMMパラメータファイルにモデルラベルおよび単語表記のフィールドを追加
                                               2013年10月20日 高木一幸
   無限ループ除去, 認識結果表示変更 2014年01月06日 緒形剛

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
#define MAXFNAMELEN     512         /* パス名,ファイル名の最大文字数 */
#define WORDNAMELEN     64               /* 単語名,HMMラベルの文字数 */





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
double viterbi(int n_frame,              /* フレーム数               */
	       float **fv,               /* 特徴ベクトル             */
	       int n_dim,                /* 特徴ベクトル次元数       */
	       int n_state,              /* HMMの状態数              */
	       int n_mix,                /* Gaussian混合数           */
	       HMM *hmm,                 /* 単語HMM                  */
	       int *bestPath,            /* 最適状態系列             */
	       char *progname);    /* このプログラムの実行ファイル名 */

void sort(double *x,
	  int *indx,
	  int n);





/*---------------------------------------------------------------------


 main


---------------------------------------------------------------------*/
int main(int argc,char *argv[])
{
  char progname[MAXFNAMELEN];      /* このプログラムの実行ファイル名 */
  int waiting;                           /* データ待ちの状態         */
  char fname_HMMlist[MAXFNAMELEN];  /* 単語HMMファイル一覧ファイル名 */
  FILE *fp_HMMlist;                   /* 単語HMMファイル一覧ファイル */
  int n_word;                            /* 単語数                   */
  int i_word;                            /* 単語カウンタ             */
  int *n_dim;                            /* 特徴ベクトル次元数       */
  int *n_state;                          /* HMMの状態数              */
  int i_state, j_state;                  /* HMMの状態カウンタ        */
  int *n_mix;                            /* HMMのGaussian混合数      */
  int i_mix;                            /* HMMのGaussian混合カウンタ */
  HMM **hmm;                             /* 単語HMMの配列            */
  char fname_HMM[MAXFNAMELEN];           /* 単語HMMファイル名        */
  char **word_name;                      /* 単語名                   */
  char **hmm_name;                       /* 単語HMMの名前            */
  int  *indx;                            /* 単語HMM番号              */
  FILE *fp_HMM;                          /* 単語HMMファイル          */
  HTKHEADER htk;                         /* HTK形式ヘッダ            */
  float **fv;                /* 特徴ベクトル保存用配列(#frame * dim) */
  int n_dim_input;                     /* 入力音声特徴ベクトル次元数 */
  int i_dim;                             /* 特徴ベクトル次元カウンタ */
  int n_frame;                           /* 入力音声フレーム数       */
  int i_frame;                         /* 入力音声フレーム数カウンタ */
  char linebuff[MAXFNAMELEN]; /*テキストファイル読み込み用1行バッファ*/
  double *lh;                            /* 尤度                     */
  int **bestPath;                        /* 最尤状態系列             */
  int result;                            /* 認識結果                 */
  int count;                             /* 認識(回数)番号           */


  /*------------------------------------------------------------
   コマンドライン数の検査
  ------------------------------------------------------------*/
  (void)strcpy(progname,argv[0]);
  if(2 != argc) {
    fprintf(stderr,"Usage: %s HMMList\n",progname);
    fprintf(stderr,"              HMMList: 単語HMM一覧\n");
    exit(EXIT_FAILURE);
  }
  (void)strcpy(fname_HMMlist,argv[1]);

 
  /*--------------------------------------

    単語HMMパラメータの読み込み

  --------------------------------------*/
  /*-------------------------------------
    単語HMMの数を数える
  -------------------------------------*/
  n_word = 0;
  if(NULL==(fp_HMMlist=fopen(fname_HMMlist,"r"))) {
    fprintf(stderr,"%s: error in opening %s\n",progname,fname_HMMlist);
    exit(EXIT_FAILURE);
  }
  while(NULL!=(fgets(linebuff,MAXFNAMELEN,fp_HMMlist))) n_word++;
  rewind(fp_HMMlist);

  /*-------------------------------------
    単語ラベル,単語名を格納する配列を生成
  -------------------------------------*/
  if(NULL==(word_name=(char**)calloc(n_word,sizeof(char*)))) {
    fprintf(stderr,"%s: error in allocating word_name\n",progname);
    exit(EXIT_FAILURE);
  }
  for(i_word=0; i_word<n_word; i_word++) {
    if(NULL==(word_name[i_word]=(char*)calloc(WORDNAMELEN+1,sizeof(char)))) {
      fprintf(stderr,"%s: error in allocating word_name[%d]\n",progname,i_word);
      exit(EXIT_FAILURE);
    }
  }
  if(NULL==(hmm_name=(char**)calloc(n_word,sizeof(char*)))) {
    fprintf(stderr,"%s: error in allocating hmm_name\n",progname);
    exit(EXIT_FAILURE);
  }
  for(i_word=0; i_word<n_word; i_word++) {
    if(NULL==(hmm_name[i_word]=(char*)calloc(WORDNAMELEN+1,sizeof(char)))) {
      fprintf(stderr,"%s: error in allocating hmm_name[%d]\n",progname,i_word);
      exit(EXIT_FAILURE);
    }
  }
  
  /*-------------------------------------
    各単語HMMの状態数を入力する
  -------------------------------------*/
  /* 特徴ベクトルの次元数を記録する配列を生成 */
  if(NULL==(n_dim=(int*)calloc(n_word,sizeof(int)))) {
    fprintf(stderr,"%s: error in allocating n_dim\n",progname);
    exit(EXIT_FAILURE);
  }
  /* 状態数を記録する配列を生成 */
  if(NULL==(n_state=(int*)calloc(n_word,sizeof(int)))) {
    fprintf(stderr,"%s: error in allocating n_state\n",progname);
    exit(EXIT_FAILURE);
  }
  /* Gaussian混合数を記録する配列を生成 */
  if(NULL==(n_mix=(int*)calloc(n_word,sizeof(int)))) {
    fprintf(stderr,"%s: error in allocating n_mix\n",progname);
    exit(EXIT_FAILURE);
  }
  /* HMMパラメータファイルから読み込む */
  if(NULL==(fp_HMMlist=fopen(fname_HMMlist,"r"))) {
    fprintf(stderr,"%s: error in opening %s\n",progname,fname_HMMlist);
    exit(EXIT_FAILURE);
  }

  for(i_word=0; i_word<n_word; i_word++) {
    /* 単語表記, モデルラベル, HMMパラメータファイル */
    if(3 != fscanf(fp_HMMlist,"%s %s %s",word_name[i_word],hmm_name[i_word],fname_HMM)) {
      fprintf(stderr,"%s: error reading word names, HMM Labels, and HMM parameter file names from %s\n",
	      progname,fname_HMMlist);
      exit(EXIT_FAILURE);
    }
    if(NULL==(fp_HMM=fopen(fname_HMM,"r"))) {
      fprintf(stderr,"%s: error in opening %s\n",progname,fname_HMM);
      exit(EXIT_FAILURE);
    }
    if(NULL==(fgets(linebuff,MAXFNAMELEN,fp_HMM))) {
      fprintf(stderr,"%s: error in fgets (dim) from %s\n",progname,fname_HMM);
      exit(EXIT_FAILURE);
    }
    sscanf(linebuff,"dim= %d\n",&n_dim[i_word]);
    if(NULL==(fgets(linebuff,MAXFNAMELEN,fp_HMM))) {
      fprintf(stderr,"%s: error in fgets (state) from %s\n",progname,fname_HMM);
      exit(EXIT_FAILURE);
    }
    sscanf(linebuff,"state= %d\n",&n_state[i_word]);
    if(NULL==(fgets(linebuff,MAXFNAMELEN,fp_HMM))) {
      fprintf(stderr,"%s: error in fgets (mix) from %s\n",progname,fname_HMM);
      exit(EXIT_FAILURE);
    }
    sscanf(linebuff,"mix= %d\n",&n_mix[i_word]);
    fclose(fp_HMM);
  }
  rewind(fp_HMMlist);


  /*-------------------------------------
    HMMパラメータ配列の生成 
  -------------------------------------*/
  /* 単語 */
  if(NULL==(hmm=(HMM**)calloc(n_word,sizeof(HMM*)))) {
    fprintf(stderr,"%s: error in allocating hmm\n",progname);
    exit(EXIT_FAILURE);
  }
  /* 単語のHMM状態 */
  for(i_word=0; i_word<n_word; i_word++) {
    if(NULL==(hmm[i_word]=(HMM*)calloc(n_state[i_word],sizeof(HMM)))) {
      fprintf(stderr,"%s: error in allocating hmm[%d]\n",progname,i_word);
      exit(EXIT_FAILURE);
    }
    for(i_state=0; i_state<n_state[i_word]; i_state++) {
      /* 状態遷移確率 a_{ij} */
      if(NULL == (hmm[i_word][i_state].a=(float*)calloc(2,sizeof(float)))) {
	fprintf(stderr,"%s: error in allocating hmm[%d][%d].a\n",progname,i_word,i_state);
	exit(1);
      }
      if(NULL == (hmm[i_word][i_state].c=(float*)calloc(n_mix[i_word],sizeof(float)))){
	fprintf(stderr,"%s: error in allocating hmm[%d][%d].c\n",progname,i_word,i_state);
	exit(1);
      }
      /* 平均値μ */
      if(NULL == (hmm[i_word][i_state].mu=(float**)calloc(n_mix[i_word],sizeof(float*)))){
	fprintf(stderr,"%s: error in allocating hmm[%d][%d].mu\n",progname,i_word,i_state);
	exit(1);
      }
      for(i_mix=0; i_mix<n_mix[i_word]; i_mix++) {
	if(NULL == (hmm[i_word][i_state].mu[i_mix]=(float*)calloc(n_dim[i_word],sizeof(float)))){
	  fprintf(stderr,"%s: error in allocating hmm[%d][%d].mu[%d]\n",progname,i_word,i_state,i_mix);
	  exit(1);
	}
      }
      /* 分散値 σ^2 */
      if(NULL == (hmm[i_word][i_state].sigma2=(float**)calloc(n_mix[i_word],sizeof(float*)))){
	fprintf(stderr,"%s: error in allocating hmm[%d][%d].sigma2\n",progname,i_word,i_state);
	exit(1);
      }
      for(i_mix=0; i_mix<n_mix[i_word]; i_mix++) {
	if(NULL == (hmm[i_word][i_state].sigma2[i_mix]=(float*)calloc(n_dim[i_word],sizeof(float)))){
	  fprintf(stderr,"%s: error in allocating hmm[%d][%d].sigma2[%d]\n",progname,i_word,i_state,i_mix);
	  exit(1);
	}
      }
    }
  }


  /*-------------------------------------------
    HMMパラメータの読み込み
  -------------------------------------------*/
  if(NULL==(fp_HMMlist=fopen(fname_HMMlist,"r"))) {
    fprintf(stderr,"%s: error in opening %s\n",progname,fname_HMMlist);
    exit(EXIT_FAILURE);
  }
  for(i_word=0; i_word<n_word; i_word++) {
    if(3 != fscanf(fp_HMMlist,"%s %s %s",word_name[i_word],hmm_name[i_word],fname_HMM)) {
      fprintf(stderr,"%s: error reading word names, HMM Labels, and HMM parameter file names from %s\n",
	      progname,fname_HMMlist);
      exit(EXIT_FAILURE);
    }
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
    for(i_state=0; i_state<n_state[i_word]; i_state++) {
      for(j_state=0; j_state<2; j_state++){
	fscanf(fp_HMM,"%e",&(hmm[i_word][i_state].a[j_state]));
      }
      /* 混合Gaussianパラメータ */
      for(i_mix=0; i_mix<n_mix[i_word]; i_mix++) {
	/* 混合係数 */
	fscanf(fp_HMM,"%e",&(hmm[i_word][i_state].c[i_mix]));
	/* 平均 */
	for(i_dim=0; i_dim<n_dim[i_word]; i_dim++) {
	  fscanf(fp_HMM,"%e",&(hmm[i_word][i_state].mu[i_mix][i_dim]));
	}
	/* 分散 */
	for(i_dim=0; i_dim<n_dim[i_word]; i_dim++) {
	  fscanf(fp_HMM,"%e",&(hmm[i_word][i_state].sigma2[i_mix][i_dim]));
	}
      }
    }
    fclose(fp_HMM);
  }
  fclose(fp_HMMlist);


  /*------------------------------------
    単語尤度の記録用配列を生成
  ------------------------------------*/
  if(NULL==(lh=(double*)calloc(n_word,sizeof(double)))) {
    fprintf(stderr,"%s: error in allocating lh\n",progname);
    exit(EXIT_FAILURE);
  }
  if(NULL==(indx=(int*)calloc(n_word,sizeof(int)))) {
    fprintf(stderr,"%s: error in allocating indx\n",progname);
    exit(EXIT_FAILURE);
  }


  /*------------------------------------
    単語認識開始
  ------------------------------------*/
  printf("Isolated Word Recogition On-The-Fly\n"); fflush(stdout);

  count = 0;
  //for(;;) {
    waiting = 1;
    do {  /* データが入ってくるまで待ち続ける */
      if(1 == fread(&htk,sizeof(HTKHEADER),1,stdin)) {
	waiting = 0;
      }
    } while(waiting);
    count++;

    /*-------------------------------------------
      認識対象の単語の特徴量ベクトルの読み込み
    -------------------------------------------*/
    n_frame = htk.nSamples;
    n_dim_input = htk.sampSize/sizeof(float);

    /* 入力音声特徴ベクトルの次元数とHMMの次元数の検査 */
    for(i_word=0; i_word<n_word; i_word++) {
      if(n_dim[i_word] != n_dim_input) {
	fprintf(stderr,"%s: HMM(%d) dimension (%d) is not equal to input feature dimension (%d)\n",
		progname,i_word,n_dim[i_word],n_dim_input);
	fflush(stderr);
	exit(EXIT_FAILURE);
      }
    }

    /* 特徴ベクトル保存用配列の生成 */
    if(NULL==(fv=(float**)calloc(n_frame,sizeof(float*)))) {
      fprintf(stderr,"%s: error in allocating fv\n",progname);
      fflush(stderr);
      exit(EXIT_FAILURE);
    }
    for(i_frame=0; i_frame<n_frame; i_frame++) {
      if(NULL==(fv[i_frame]=(float*)calloc(n_dim_input,sizeof(float)))) {
	fprintf(stderr,"%s: error in allocating fv[%d]\n",progname,i_frame);
	fflush(stderr);
	exit(EXIT_FAILURE);
      }
    }
    /* 特徴量ベクトルの読み込み */
    for(i_frame=0; i_frame<n_frame; i_frame++) {
      if(n_dim_input != fread(fv[i_frame],sizeof(float),n_dim_input,stdin)) {
	fprintf(stderr,"%s: error in reading feature vector of frame %d from stdin\n",
		progname,i_frame);
	fflush(stderr);
	exit(EXIT_FAILURE);
      }
    }

    /* 最尤状態系列用配列を生成          */
    if(NULL==(bestPath=(int**)calloc(n_word,sizeof(int*)))) {
      fprintf(stderr,"%s: error in allocating bestPath\n",progname);
      fflush(stderr);
      exit(EXIT_FAILURE);
    }
    for(i_word=0; i_word<n_word; i_word++) {
      if(NULL==(bestPath[i_word]=(int*)calloc(n_frame,sizeof(int)))) {
	fprintf(stderr,"%s: error in allocating bestPath[%d]\n",progname,i_word);
	fflush(stderr);
	exit(EXIT_FAILURE);
      }
    }

    /* Viterbiデコーディング */
    for(i_word=0; i_word<n_word; i_word++) {
      lh[i_word] = viterbi(n_frame,fv,n_dim[i_word],n_state[i_word],n_mix[i_word],hmm[i_word],bestPath[i_word],progname);
    }
    
    /* 尤度の降順に整列 */
    for(i_word=0; i_word<n_word; i_word++) indx[i_word] = i_word;
    sort(lh,indx,n_word);

    /* 認識結果を表示 */
    /* 1位 */
    //printf("[%d] %s (%s)= %f\n",count,word_name[indx[0]],hmm_name[indx[0]],lh[0]);
    printf("--------------------\n"); //追加(緒形)
    printf("rank word (kana)= log-likelihood\n"); //追加(緒形)
    printf("--------------------\n"); //追加(緒形)
    printf("%2d. %s (%s)= %f\n",1,word_name[indx[0]],hmm_name[indx[0]],lh[0]); //追加(緒形)

    /* 2位以下 */
    for(i_word=1; i_word<n_word; i_word++) 
      printf("%2d. %s (%s)= %f\n",
	     (i_word+1),word_name[indx[i_word]],hmm_name[indx[i_word]],lh[i_word]);
    fflush(stdout);
    printf("\n"); //追加(緒形)
    //}
    return 0;
}





/*---------------------------------------------------------------------

 sort

---------------------------------------------------------------------*/
void sort(double *x,
	  int *indx,
	  int n)
{
  int i,j;
  double t;
  int k;


  for(i=0; i<n-1; i++) {
    for(j=i+1; j<n; j++)
      if(x[i] < x[j]) {
	t = x[i];
	x[i] = x[j];
	x[j] = t;
	k = indx[i];
	indx[i] = indx[j];
	indx[j] = k;
      }
  }
}
