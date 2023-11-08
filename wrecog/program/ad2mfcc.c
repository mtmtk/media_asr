/**********************************************************************

  ad2mfcc: メル周波数ケプストラム係数(MFCC)分析


                                               2013年10月17日 高木一幸
                                         改訂  2013年11月26日 高木一幸

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
#define NUMCHANS     28                  /* フィルタバンクチャネル数 */
#define NUMCEPS      20                  /* フィルタバンクチャネル数 */





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




/*---------------------------------------------------------------------

  function prototypes

---------------------------------------------------------------------*/
void ad2mfcc(int n_frame,                /* フレーム数               */
	     int n_shift,                /* シフト幅 [サンプル値]    */
	     short int *ad,              /* 音声サンプル(実数)       */
	     int n_sample,               /* 音声サンプル数           */
	     float **mfcc,               /* MFCC                     */
	     char *progname);            /* プログラム名             */

void ad2fb(int n_frame,                  /* フレーム数               */
	   int n_shift,                  /* シフト幅 [サンプル値]    */
	   short int *ad,                /* 音声サンプル(実数)       */
	   int n_sample,                 /* 音声サンプル数           */
	   float **fb,                   /* フィルタバンク出力値     */
	   char *progname);              /* プログラム名             */

void fb2mfcc(int n_frame,                /* フレーム数               */
	     float **fb,                 /* フィルタバンク           */
	     float **mfcc,               /* MFCC                     */
	     int P,                      /* フィルタバンクの次元     */
	     int Q);                     /* MFCCの次元               */





/*---------------------------------------------------------------------

  ad2mfcc: 音声波形からMFCC時系列を計算する

---------------------------------------------------------------------*/
void ad2mfcc(int n_frame,                /* フレーム数               */
	     int n_shift,                /* シフト幅 [サンプル値]    */
	     short int *ad,              /* 音声サンプル(整数)       */
	     int n_sample,               /* 音声サンプル数           */
	     float **mfcc,               /* MFCC                     */
	     char *progname              /* プログラム名             */
)
{
  int i_frame;                           /* フレームindex            */ 
  int ist;                               /* 分析開始サンプル番号     */
  float **fb;                            /* フィルタバンク出力値     */
  int i;
  double mean;
  double sqsum;
  double sd;


  /*-------------------------------------------
    フィルタバンク出力値を格納する配列を生成
  -------------------------------------------*/
  if(NULL==(fb=(float**)calloc(n_frame,sizeof(float*)))) {
    fprintf(stderr,"%s: error in allocating fb\n",progname);
    exit(EXIT_FAILURE);
  }
  for(i_frame=0; i_frame<n_frame; i_frame++) {
    if(NULL==(fb[i_frame]=(float*)calloc(NUMCHANS,sizeof(float)))) {
      fprintf(stderr,"%s: error in allocating fb[%d]\n",progname,i_frame);
      exit(EXIT_FAILURE);
    }
  }

  /* 音声波形 x からフィルタバンク出力値 fb を計算 */
  ad2fb(n_frame,n_shift,ad,n_sample,fb,progname);

  /* フィルタバンク出力値 fb からMFCCを計算 */
  fb2mfcc(n_frame,fb,mfcc,NUMCHANS,NUMCEPS);

  /* Cepstral Mean and Variance Normalization (CMVN) */
  for(i=0; i<NUMCEPS; i++) {
    /* 平均値(mean)の計算 */
    mean = 0.0;
    for(i_frame=0; i_frame<n_frame; i_frame++) {
      mean += mfcc[i_frame][i];
    }
    mean /= (float)n_frame;
    /* 標準偏差(sd)の計算 */
    sqsum = 0.0;
    for(i_frame=0; i_frame<n_frame; i_frame++)
      sqsum += (mfcc[i_frame][i]-mean)*(mfcc[i_frame][i]-mean);
    sd = sqrt(sqsum/(float)n_frame);
    /* CMVN */
    for(i_frame=0; i_frame<n_frame; i_frame++)
      mfcc[i_frame][i] = (mfcc[i_frame][i]-mean)/sd;
  }
    
  /* メモリを開放 */
  for(i_frame=0; i_frame<n_frame; i_frame++) 
    free(fb[i_frame]);
  free(fb);

  return;
}
