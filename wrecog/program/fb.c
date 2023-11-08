/**********************************************************************

  fb: フィルタバンク分析


  使用法
    % fb file.wav file.fb

  入力
      file.wav: 入力信号波形データファイル
               サンプリング周波数は16kHz, 振幅値は16bit符合付整数

  出力
      file.fb: フィルタバンク出力ファイル(gnuplot描画用のテキスト形式)

  作成履歴
     初版                                       2013年10月05日 高木一幸
     各チャネルの対応周波数を訂正                  2013年10月19日 高木一幸
     wav形式ファイル用に改造                      2014年10月16日 高木一幸

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
#define PREEMCOEF    0.97                /* 高域強調係数             */
#define NYQUISTFREQ  8000                /* ナイキスト周波数 [Hz]    */
#define WINDOW_WIDTH 32                  /* 窓幅 [ms]                */   
#define WINDOW_SHIFT 10                  /* 窓シフト [ms]            */
#define NUMCHANS     28                  /* フィルタバンクチャネル数 */
#define SWAP(a,b)    tempr=(a);(a)=(b);(b)=tempr /* aとbの値の交換   */





/*---------------------------------------------------------------------

 data definitions

---------------------------------------------------------------------*/
/* WAV形式ヘッダ(linear PCM) */
typedef struct wavheader {
  char riff[4];                          /* RIFF形式の識別子 'RIFF   */
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





/*---------------------------------------------------------------------

  function prototypes

---------------------------------------------------------------------*/
void ad2fb(int n_frame,                  /* フレーム数               */
	   int n_shift,                  /* シフト幅 [サンプル値]    */
	   short int *ad,                /* 音声サンプル(整数)       */
	   int n_sample,                 /* 音声サンプル数           */
	   float **fb,                   /* フィルタバンク出力値     */
	   char *progname);              /* プログラム名             */

float mel(float f);                      /* 周波数 [Hz]              */

float lnr(float m);                      /* メル周波数 [Mel]         */




/*---------------------------------------------------------------------

 usage: 使用方法

---------------------------------------------------------------------*/
void usage(char *progname) {
  fprintf(stderr, "\n");
  fprintf(stderr, " %s - フィルタバンク分析\n",progname);
  fprintf(stderr, "\n");
  fprintf(stderr, "  使用法:\n");
  fprintf(stderr, "      %s file.wav file.fb\n",progname);
  fprintf(stderr, "  入力\n");
  fprintf(stderr, "      file.wav: 入力信号波形データファイル\n");
  fprintf(stderr, "               サンプリング周波数は16kHz, 振幅値は16bit符合付整数\n");
  fprintf(stderr, "  出力\n");
  fprintf(stderr, "      file.fb: フィルタバンク出力ファイル(gnuplot描画用のテキスト形式)\n\n");
 }





/*---------------------------------------------------------------------

 メインプログラム

---------------------------------------------------------------------*/
int main(int argc, char *argv[]) {
  char progname[MAXFNAMELEN];            /* このプログラムのコマンド名  */
  char fname_wav[MAXFNAMELEN];           /* 音声波形ファイル名         */
  WAVHEADER wavhdr;                      /* WAV形式ヘッダ(linear PCM) */
  char fname_fb[MAXFNAMELEN];            /* フィルタバンクファイル名   */
  FILE *fp_wav;                          /* 音声波形ファイル         */
  FILE *fp_fb;                           /* フィルタバンクファイル   */
  long int filesize;                     /* ファイルサイズ [byte]    */
  int n_sample;                          /* 音声サンプル数           */
  int i_sample;                          /* 音声サンプルindex        */
  short int *ad;                         /* 音声サンプル(整数)       */
  int n_window;                          /* 窓幅 [サンプル値]        */
  int n_shift;                           /* シフト幅 [サンプル値]    */
  int n_frame;                           /* フレーム数               */
  int i_frame;                           /* フレームindex            */
  int ist;                               /* 分析開始サンプル番号     */
  int i;
  int l;                                 /* チャンネルindex          */
  float **fb;                            /* フィルタバンク出力値     */
  float t;                               /* フレーム中心時間[ms]     */
  float f;                               /* ビン周波数[Hz]           */



  /*-------------------------------------
    コマンドラインオプションの処理
  -------------------------------------*/
  strcpy(progname,argv[0]);
  if(3 > argc) {
    usage(progname);
    exit(EXIT_FAILURE);
  }
  strcpy(progname,argv[0]);
  strcpy(fname_wav,argv[1]);
  strcpy(fname_fb,argv[2]);

  
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
  
  /*-------------------------------------
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
   フィルタバンク分析
  -------------------------------------*/
  n_window = WINDOW_WIDTH*SFREQ;
  n_shift = WINDOW_SHIFT*SFREQ;
  n_frame = (int)((float)(n_sample-(n_window-n_shift))/(float)n_shift);
  
  /* フィルタバンク出力値を格納する配列をa生成 */
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

  /* 音声波形 ad からフィルタバンク出力値 fb を計算 */
  ad2fb(n_frame,n_shift,ad,n_sample,fb,progname);

  /* フィルタバンクファイルを開く */
  if(NULL== (fp_fb = fopen(fname_fb,"wt"))) {
    printf("%s: error in opening %s\n",progname,fname_fb);
    exit(1);
  }
  /* フィルタバンクファイルに保存(gnuplot描画用のテキスト形式) */
  for(i_frame=0; i_frame<n_frame; i_frame++) {
    for(l=0; l<NUMCHANS; l++) {
      t = (i_frame+1)*WINDOW_SHIFT;
      fprintf(fp_fb,"%.1f\t%d\t%e\n",t,l+1,fb[i_frame][l]);
    }
    fprintf(fp_fb,"\n");
  }

  /* フィルタバンクファイルを閉じる */
  fclose(fp_fb);

  /* 配列を開放  */
  free(ad);
  for(i_frame=0; i_frame<n_frame; i_frame++) free(fb[i_frame]);
  free(fb);

  exit(EXIT_SUCCESS);
}
