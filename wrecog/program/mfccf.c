/**********************************************************************

  mfccf: メル周波数ケプストラム係数(MFCC)分析〜ファイル入出力版〜


  使用法
    % mfccf file.wav file.mfcc

  入力
      file.wav: 入力信号波形データファイル
                サンプリング周波数は16kHz, 振幅値は16bit符合付整数

  出力
      file.mfcc: MFCCファイル

  作成履歴
     初版                                     2013年10月06日 高木一幸
     wav形式ファイル用に改造                  2014年10月16日 高木一幸

**********************************************************************/
/*---------------------------------------------------------------------

  include files

---------------------------------------------------------------------*/
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>





/*---------------------------------------------------------------------

  macros

---------------------------------------------------------------------*/
#define MAXFNAMELEN  256                 /* ファイル名最大長         */
#define SFREQ        16                  /* サンプリング周波数 [kHz] */
#define WINDOW_WIDTH 32                  /* 窓幅 [ms]                */   
#define WINDOW_SHIFT 10                  /* 窓シフト [ms]            */
#define NUMCEPS      20                  /* フィルタバンクチャネル数 */





/*---------------------------------------------------------------------

 data definitions

---------------------------------------------------------------------*/
/* WAV形式ヘッダ(linear PCM) */
typedef struct wavheader {
  char riff[4];                          /* RIFF形式の識別子 'RIFF'  */
  int filesize;                          /* ファイルサイズ(4 byte)   */
  char wave[4];                      /* RIFFの種類を表す識別子 'WAVE' */
  char fmt[4];                           /* フォーマットの定義        */
  int fmtchunksize;                    /* フォーマットチャンクのbyte数 */
  short int formatid; /* フォーマットID(リニアPCMならば16(10 00 00 00)) */
  short int channels;              /* チャネル数(モノラルならば1(01 00) */
  int samprate;                          /* サンプリングレート        */
  int datarate;                          /* データ速度(byte/sec)     */
  short int blocksize;                   /* ブロックサイズ           */
  short int bitpersample;                /* サンプルあたりのビット数 */
  char data[4];                          /* 'data'                   */
  int datasize;                          /* 波形データのバイト数     */
} WAVHEADER;          

/* HTK形式ヘッダ */
typedef struct htktype {
  int   nSamples;                        /* サンプル数               */
  int   sampPeriod;                      /* サンプル間隔             */
  short sampSize;                        /* サンプルサイズ[byte]     */
  short parmKind;                        /* パラメータの種類         */
} HTKHEADER;          





/*---------------------------------------------------------------------

  function prototypes

---------------------------------------------------------------------*/
void ad2mfcc(int n_frame,                /* フレーム数               */
	     int n_shift,                /* シフト幅 [サンプル値]    */
	     short int *ad,              /* 音声サンプル(整数)       */
	     int n_sample,               /* 音声サンプル数           */
	     float **mfcc,               /* MFCC                     */
	     char *progname);            /* プログラム名             */





/*---------------------------------------------------------------------

 usage: 使用方法

---------------------------------------------------------------------*/
void usage(char *progname) {
  fprintf(stderr, "\n");
  fprintf(stderr, " %s - メル周波数ケプストラム係数(MFCC)分析\n",progname);
  fprintf(stderr, "\n");
  fprintf(stderr, "  使用法:\n");
  fprintf(stderr, "      %s file.wav file.mfcc\n",progname);
  fprintf(stderr, "  入力\n");
  fprintf(stderr, "      file.wav: 入力信号波形データファイル\n");
  fprintf(stderr, "                サンプリング周波数は16kHz, 振幅値は16bit符合付整数\n");
  fprintf(stderr, "  出力\n");
  fprintf(stderr, "      file.fb: MFCCファイル\n\n");
 }





/*---------------------------------------------------------------------

 メインプログラム

---------------------------------------------------------------------*/
int main(int argc, char *argv[]) {
  char progname[MAXFNAMELEN];            /* このプログラムのコマンド名 */
  char fname_wav[MAXFNAMELEN];           /* 音声波形ファイル名       */
  char fname_mfcc[MAXFNAMELEN];          /* MFCCファイル名           */
  FILE *fp_wav;                          /* 音声波形ファイル         */
  FILE *fp_mfcc;                         /* フィルタバンクファイル   */
  WAVHEADER wavhdr;                      /* WAV形式ヘッダ(linear PCM) */
  long int filesize;                     /* ファイルサイズ [byte]    */
  int n_sample;                          /* 音声サンプル数           */
  int i_sample, i_offset;                /* 音声サンプルindex        */
  short int *ad;                         /* 音声サンプル(整数)       */
  int n_window;                          /* 窓幅 [サンプル値]        */
  int n_shift;                           /* シフト幅 [サンプル値]    */
  int n_frame;                           /* フレーム数               */
  int i_frame;                           /* フレームindex            */
  int i, j, k, l, m;
  HTKHEADER htk;                         /* HTK形式ヘッダ            */
  float **mfcc;                          /* MFCC                     */



  /*-------------------------------------
    コマンドラインオプションの処理
  -------------------------------------*/
  strcpy(progname,argv[0]);
  if(3 > argc) {
    usage(progname);
    exit(EXIT_FAILURE);
  }
  strcpy(fname_wav,argv[1]); 
  strcpy(fname_mfcc,argv[2]); 

  /*-------------------------------------
    音声ファイルを開く
  -------------------------------------*/
  if(NULL== (fp_wav = fopen(fname_wav,"rb"))) {
    printf("%s: error in opening %s\n",progname,fname_wav);
    exit(EXIT_FAILURE);
  }
  if(1 != fread(&wavhdr,sizeof(WAVHEADER),1,fp_wav)) { /* WAV形式ヘッダ(linear PCM)を読み込む */
    printf("%s: error in fread (WAVHEADER) on %s\n",progname,fname_wav);
    exit(EXIT_FAILURE);
  }
  n_sample = wavhdr.datasize/sizeof(short int); /* 音声サンプル数 */
  
  /*-----------------------------------
    音声データを格納する配列を生成
  -------------------------------------*/
  if(NULL==(ad=(short int *)calloc(n_sample,sizeof(short int)))) {
    fprintf(stderr,"%s: error in allocating ad\n",progname);
    exit(EXIT_FAILURE);
  }

  /*------------------------------------
    音声データを読み込む
  -------------------------------------*/ 
  if(n_sample != fread(ad,sizeof(short int),n_sample,fp_wav)) {
    fprintf(stderr,"%s: error in reading ad from file %s\n",progname,fname_wav);
    exit(EXIT_FAILURE);
  }
  fclose(fp_wav);                         /* 音声ファイルを閉じる     */

  /*------------------------------------
   MFCC分析
  -------------------------------------*/
  n_window = WINDOW_WIDTH*SFREQ;
  n_shift = WINDOW_SHIFT*SFREQ;
  n_frame = (int)((float)(n_sample-(n_window-n_shift))/(float)n_shift);
  
  /* MFCCを格納する配列を生成 */
  if(NULL==(mfcc=(float**)calloc(n_frame,sizeof(float*)))) {
    fprintf(stderr,"%s: error in allocating mfcc\n",progname);
    exit(EXIT_FAILURE);
  }
  for(i_frame=0; i_frame<n_frame; i_frame++) {
    if(NULL==(mfcc[i_frame]=(float*)calloc(NUMCEPS,sizeof(float)))) {
      fprintf(stderr,"%s: error in allocating mfcc[%d]\n",progname,i_frame);
      exit(EXIT_FAILURE);
    }
  }

  /* 音声波形 ad からMFCC時系列 mfcc を計算 */
  ad2mfcc(n_frame,n_shift,ad,n_sample,mfcc,progname);

  /* MFCCファイルを開く */
  if(NULL== (fp_mfcc = fopen(fname_mfcc,"wt"))) {
    printf("%s: error in opening %s\n",progname,fname_mfcc);
    exit(EXIT_FAILURE);
  }

  /* HTK形式ヘッダを書く */
  htk.nSamples = n_frame;
  htk.sampPeriod = WINDOW_SHIFT*10000;
  htk.sampSize = NUMCEPS*sizeof(float);
  htk.parmKind = 0; /* don not care */
  if(1 != fwrite(&htk,sizeof(HTKHEADER),1,fp_mfcc)) {
    fprintf(stderr,"%s: error in writing data to file %s\n",progname,fname_mfcc);
    exit(EXIT_FAILURE);
  }

  /* MFCCをファイルに保存 */
  for(i_frame=0; i_frame<n_frame; i_frame++) {
    if(NUMCEPS != fwrite(mfcc[i_frame],sizeof(float),NUMCEPS,fp_mfcc)) {
      fprintf(stderr,"%s: error in writing MFCC to file %s\n",progname,fname_mfcc);
      exit(EXIT_FAILURE);
    }
  }
  
  /* MFCCファイルを閉じる */
  fclose(fp_mfcc);

  /* メモリを開放 */
  free(ad);
  for(i_frame=0; i_frame<n_frame; i_frame++) free(mfcc[i_frame]);
  free(mfcc);

  exit(EXIT_SUCCESS);
}
