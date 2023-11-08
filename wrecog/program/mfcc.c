/**********************************************************************

  mfcc: メル周波数ケプストラム係数(MFCC)分析〜ストリーム入出力版〜


  使用法
    %  source | mfcc 

  入力
      source: 音声波形をstdoutに出力し続けるアプリケーション
             サンプリング周波数は16kHz, 振幅値は16bit符合付整数

  出力
      stdout: MFCC


  作成履歴
  初版                                          2013年10月06日 高木一幸
  無限ループ除去                                 2014年01月06日 緒形剛
  入力音声データをプロセス番号を名前とするWAVファイルに保存
                                               2014年10月16日 高木一幸

**********************************************************************/
/*---------------------------------------------------------------------

  include files

---------------------------------------------------------------------*/
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>





/*---------------------------------------------------------------------

  macros

---------------------------------------------------------------------*/
#define MODE_STREAM  0                   /* 入出力モード(stream)     */
#define MODE_FILE    1                   /* 入出力モード(file)       */
#define MAXFNAMELEN  1024                /* ファイル名最大長         */
#define SFREQ        16                  /* サンプリング周波数 [kHz] */
#define SEGDUR       20                  /* セグメント長[ms]         */
#define PREEMCOEF    0.97                /* 高域強調係数             */
#define NYQUISTFREQ  8000                /* ナイキスト周波数 [Hz]    */
#define WINDOW_WIDTH 32                  /* 窓幅 [ms]                */   
#define WINDOW_SHIFT 10                  /* 窓シフト [ms]            */
#define N_DFT        512                 /* DFT点数                  */
#define N_FREQ       256                 /* スペクトル点数           */
#define NUMCHANS     28                  /* フィルタバンクチャネル数 */
#define NUMCEPS      20                  /* フィルタバンクチャネル数 */
#define SWAP(a,b)    tempr=(a);(a)=(b);(b)=tempr /* aとbの値の交換   */
#define NSEGSAMPLE   (SFREQ*SEGDUR) /* 1セングメントの音声サンプル数 */
#define ENDSIGNAL    SHRT_MAX /*この数値がENDSIGNALLEN個で音声区間検出終了*/
#define ENDSIGNALLEN (SFREQ*SEGDUR)      /* 終了信号の個数           */





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


/* 入力音声データのバッファ */
typedef struct segchain {
  struct segchain *next;                 /* 次のセグメント           */
  short int *seg;                        /* 音声サンプル             */
} ADBUFF;


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
int is_end_of_signal(ADBUFF *p);         /* 音声サンプル(バッファ)   */

void ad2mfcc(int n_frame,                /* フレーム数               */
	     int n_shift,                /* シフト幅 [サンプル値]    */
	     short int *ad,              /* 音声サンプル(整数)       */
	     int n_sample,               /* 音声サンプル数           */
	     float **mfcc,               /* MFCC                     */
	     char *progname);            /* プログラム名             */





/*---------------------------------------------------------------------

 メインプログラム

---------------------------------------------------------------------*/
int main(int argc, char *argv[]) {
  char progname[MAXFNAMELEN];          /* このプログラムのコマンド名 */
  int waiting;                           /* データ待ちの状態         */
  short int buf[NSEGSAMPLE];             /* データ待ち受け用バッファ */
  int n_sample;                          /* 音声サンプル数           */
  int i_segsample, i_sample, i_offset;   /* 音声サンプルindex        */
  ADBUFF *head, *next, *cur, *tail;      /* 入力バッファへのポインタ */
  ADBUFF *del, *tmp;             /* 入力バッファへのポインタ(削除用) */
  short int *ad;                         /* 音声サンプル(整数)       */
  int n_window;                          /* 窓幅 [サンプル値]        */
  int n_shift;                           /* シフト幅 [サンプル値]    */
  int n_frame;                           /* フレーム数               */
  int i_frame;                           /* フレームindex            */
  int ist;                               /* 分析開始サンプル番号     */
  int i, j, k, l, m;
  HTKHEADER htk;                         /* HTK形式ヘッダ            */
  float **mfcc;                          /* MFCC                     */
  char fname_wav[MAXFNAMELEN];           /* WAVファイル名             */
  FILE *fp_wav;                          /* WAVファイル              */
  WAVHEADER wavhdr;                      /* WAV形式ヘッダ(linear PCM) */



  strcpy(progname,argv[0]);

  //for(;;) {
    waiting = 1;
    do {  /* データが入ってくるまで待ち続ける */
      if(NSEGSAMPLE == fread(buf,sizeof(short int),NSEGSAMPLE,stdin)) {
	waiting = 0;
      }
    } while(waiting);
    
   /* データが入ってきたらそれを先ずバッファの先頭に保存 */
    if(NULL==(head=(ADBUFF*)calloc(1,sizeof(ADBUFF)))) {
      fprintf(stderr,"%s: error in allocating waveform buffer\n",progname);
      exit(EXIT_FAILURE);
    }
    if(NULL==(head->seg=(short int*)calloc(NSEGSAMPLE,sizeof(short int)))) {
      fprintf(stderr,"%s: error in allocating waveform buffer(seg)\n",progname);
      exit(EXIT_FAILURE);
    }
    head->next = NULL;
    for(i_segsample=0; i_segsample<NSEGSAMPLE; i_segsample++) {
      head->seg[i_segsample] = buf[i_segsample];
    }
    n_sample = NSEGSAMPLE;
    cur = head;

    do {
      /* 次のデータのためのバッファを生成 */
      if(NULL==(cur->next=(ADBUFF*)calloc(1,sizeof(ADBUFF)))) {
	fprintf(stderr,"%s: error in allocating new waveform buffer\n",progname);
	exit(EXIT_FAILURE);
      }
      cur = cur->next;
      if(NULL==(cur->seg=(short int*)calloc(NSEGSAMPLE,sizeof(short int)))) {
	fprintf(stderr,"%s: error in allocating adbuff->samples\n",progname);
	exit(EXIT_FAILURE);
      }
      cur->next = NULL;

      /* 次のデータを読む */
      if(NSEGSAMPLE != fread(cur->seg,sizeof(short int),NSEGSAMPLE,stdin)) {
	fprintf(stderr,"%s: error in reading data from stdin\n",progname);
	exit(EXIT_FAILURE);
      }
      n_sample += NSEGSAMPLE;
      tail = cur;
    } while(! is_end_of_signal(cur));
    n_sample -= NSEGSAMPLE; /* 最後のバッファのデータは捨てる */


    /* バッファに溜った音声信号を1本の配列にコピー */
    /* 音声データを格納する配列を生成 */
    if(NULL==(ad=(short int *)calloc(n_sample,sizeof(short int)))) {
      fprintf(stderr,"%s: error in allocating ad\n",progname);
      exit(EXIT_FAILURE);
    }
    
    cur = head;
    i_offset = 0;
    while(! is_end_of_signal(cur)) { /* 終了信号が来るまでコピー */
      for(i_sample=0; i_sample<NSEGSAMPLE; i_sample++)
	ad[i_offset+i_sample] = cur->seg[i_sample];
      i_offset += NSEGSAMPLE;
      cur = cur->next;
    }

    /*------------------------------------
      入力音声をプロセス番号をファイル名とする
      ファイルに保存する
    -------------------------------------*/
    wavhdr.riff[0] = 'R'; 
    wavhdr.riff[1] = 'I'; 
    wavhdr.riff[2] = 'F'; 
    wavhdr.riff[3] = 'F'; 
    wavhdr.filesize = sizeof(WAVHEADER)+n_sample*sizeof(short int)-8;
    wavhdr.wave[0] = 'W';
    wavhdr.wave[1] = 'A';
    wavhdr.wave[2] = 'V';
    wavhdr.wave[3] = 'E';
    wavhdr.fmt[0] = 'f';
    wavhdr.fmt[1] = 'm';
    wavhdr.fmt[2] = 't';
    wavhdr.fmt[3] = ' ';
    wavhdr.fmtchunksize = 16;
    wavhdr.formatid = 1;
    wavhdr.channels = 1;
    wavhdr.samprate = 16000;
    wavhdr.datarate = 32000;
    wavhdr.blocksize = 2;
    wavhdr.bitpersample = 16;
    wavhdr.data[0] = 'd';
    wavhdr.data[1] = 'a';
    wavhdr.data[2] = 't';
    wavhdr.data[3] = 'a';
    wavhdr.datasize = n_sample*sizeof(short int);
    sprintf(fname_wav,"vadwav/%d.wav",getpid());
    /* WAVファイルを開く */
    if(NULL== (fp_wav = fopen(fname_wav,"wb"))) {
      fprintf(stderr,"%s: error in opening %s\n",progname,fname_wav);
      exit(EXIT_FAILURE);
    }
    /* WAV形式ヘッダを書く */
    if(1 != fwrite(&wavhdr,sizeof(WAVHEADER),1,fp_wav)) {
      fprintf(stderr,"%s: error in writing header to file %s\n",progname,fname_wav);
      exit(EXIT_FAILURE);
    }
    /* 音声データを書く */
    if(n_sample != fwrite(ad,sizeof(short int),n_sample,fp_wav)) {
      fprintf(stderr,"%s: error in writing wave data to file %s\n",progname,fname_wav);
      exit(EXIT_FAILURE);
    }
    /* WAVファイルを閉じる */
    fclose(fp_wav);
    fprintf(stderr,"vad_file= %s\n",fname_wav);
 
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
    
    htk.nSamples = n_frame;
    htk.sampPeriod = WINDOW_SHIFT*10000;
    htk.sampSize = NUMCEPS*sizeof(float);
    htk.parmKind = 0; /* do not care */
    /* HTK形式ヘッダを書く */
    if(1 != fwrite(&htk,sizeof(HTKHEADER),1,stdout)) {
      fprintf(stderr,"%s: error in writing data to stdout\n",progname);
      exit(EXIT_FAILURE);
    }
    /* MFCCデータを書き出す */
    for(i_frame=0; i_frame<n_frame; i_frame++) {
      if(NUMCEPS != fwrite(mfcc[i_frame],sizeof(float),NUMCEPS,stdout)) {
	fprintf(stderr,"%s: error in writing MFCC to stdout\n",progname);
	exit(EXIT_FAILURE);
      }
    }
    /* メモリを開放 */
    free(ad);
    for(i_frame=0; i_frame<n_frame; i_frame++) free(mfcc[i_frame]);
    free(mfcc);
  
    if(head) {
      tmp = head;
      while(tmp) {
	del = tmp;
	tmp = tmp->next;
	free(del->seg);
	free(del);
      }
    }
    head = tail = NULL;
    //}
    return 0;
}





/*---------------------------------------------------------------------

  is_end_of_signal: 音声信号の終りを判断

      p->segの最初からENDSIGNALLEN個のENDSIGNALが連続して入っていれば
      音声区間検出信号と判断する。


---------------------------------------------------------------------*/
int is_end_of_signal(ADBUFF *p)
{
  int i_sample;                          /* 音声サンプルindex        */
  int n_endsignal;                       /* ENDSIGNALの個数          */

  
  n_endsignal = 0;
  for(i_sample=0; i_sample<NSEGSAMPLE; i_sample++)
    if(ENDSIGNAL == p->seg[i_sample]) n_endsignal++;

  if(ENDSIGNALLEN == n_endsignal)
    return 1;
  else
    return 0;			
}
