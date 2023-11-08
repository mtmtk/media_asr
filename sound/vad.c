/**********************************************************************

  vad: 音声区間の検出
         標準入力からの信号を一定幅に区切ってセグメントとする。
         閾値を超えるセグメントを音声区間のセグメントとして出力する。
         音声区間中で閾値を下回るセグメントについては,その数が
         連続して5個以下であれば,音声区間とみなす。連続して5個を
         超えたとき,音声区間が終了したと判断し,出力を打ち切る。

  使用法
    vad -p th [-a [name]][-h] < file.ad 

  入力
               pth: パワーの閾値 [dB]
           file.ad: 入力信号波形データファイルあるいはストリーム
                    サンプリング周波数は16kHz, 振幅値は16bit符合付整数
                -a: このオプションを指定すると検出した音声区間をファイ
                    ルに保存。ファイル名はx0001,x0002,... となる。
              name: 生成されるファイルの接頭辞を指定。
                    ファイル名は"x"の代りに指定された接頭辞が使われる。
  出力
       -a が指定されていない場合(省略時動作)
            stdout: 音声区間波形データ(データ形式は入力と同一)
       -a が指定された場合
            nameにaa, ab,...がつけられた名前のファイル

  変更履歴
                     
       初版                                   2013年09月23日 高木一幸
       ストリーム出力の場合の検出終了信号     2013年10月06日 高木一幸
       出力ファイルに拡張子 "ad" を付ける     2013年10月19日 高木一幸
       無限ループ除去                         2014年01月06日 緒形剛
       バッファのサイズを5から10に変更        2014年11月04日 高木一幸

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
#define MAXFNAMELEN  256                 /* ファイル名文字列最大長   */
#define SFREQ        16                  /* サンプリング周波数[kHz]  */
#define SEGDUR       20                  /* セグメント長[ms]         */
#define BUFFSIZE     10                /* バッファサイズ[セグメント] */
#define MAXLOWSEG    5 /* 音声区間中で閾値を下回るセグメントの最大数 */
#define INSPEECH     1                   /* 音声区間にいる           */
#define NOTINSPEECH  0                   /* 音声区間にいない         */
#define WRITE_TO_STDOUT 0                /* 動作モード(stdout出力)   */
#define WRITE_TO_FILE   1                /* 動作モード(file出力)     */
#define NSEGSAMPLE   (SFREQ*SEGDUR) /* 1セングメントの音声サンプル数 */
#define ENDSIGNAL    SHRT_MAX /*この数値がENDSIGNALLEN個で音声区間検出終了*/
#define ENDSIGNALLEN  (SFREQ*SEGDUR)     /* 終了信号の個数           */




/*---------------------------------------------------------------------

 data definitions

---------------------------------------------------------------------*/
/* 入力音声データのバッファ */
typedef struct segchain {
  struct segchain *next;                 /* 次のセグメント           */
  short int *seg;                        /* 音声サンプル             */
} ADBUFF;





/*---------------------------------------------------------------------

  function prototypes

---------------------------------------------------------------------*/
float short_time_energy(short int *x,
			int n);





/*---------------------------------------------------------------------

 usage: 使用方法

---------------------------------------------------------------------*/
void usage(char *progname) {
  fprintf(stderr, "\n");
  fprintf(stderr, " %s - 音声区間の検出\n",progname);
  fprintf(stderr, "\n");
  fprintf(stderr, "  使用法:\n");
  fprintf(stderr, "       %s -p th [-a [name]][-h] < file.ad\n",progname);
  fprintf(stderr, "  入力:\n");
  fprintf(stderr, "         pth: パワーの閾値 [dB]\n");
  fprintf(stderr, "     file.ad: 入力信号波形データファイルあるいはストリーム\n");
  fprintf(stderr, "              サンプリング周波数は16kHz, 振幅値は16bit符合付整数\n");
  fprintf(stderr, "               -h: 使用法を表示\n");
  fprintf(stderr, "              -a: このオプションを指定すると検出した音声区間をファイ\n");
  fprintf(stderr, "                  ルに保存。ファイル名はx0001.ad,x0002.ad,... となる。\n");
  fprintf(stderr, "            name: 生成されるファイルの接頭辞を指定。\n");
  fprintf(stderr, "                  ファイル名は\"x\"の代りに指定された接頭辞が使われる。\n");
  fprintf(stderr, "  出力:\n");
  fprintf(stderr, "      -a が指定されていない場合(省略時動作)\n");
  fprintf(stderr, "          stdout: 音声区間波形データあるいはストリーム\n");
  fprintf(stderr, "                  データ形式は入力と同一\n");
  fprintf(stderr, "     -a が指定された場合\n");
  fprintf(stderr, "          nameにaa, ab,...がつけられた名前のファイル\n");
  fprintf(stderr, "\n");
}





/*---------------------------------------------------------------------

 メインプログラム

---------------------------------------------------------------------*/
int main(int argc, char *argv[]) {
  char progname[MAXFNAMELEN];          /* このプログラムのコマンド名 */
  char *s, c;                  /* コマンドラインオプション処理用変数 */
  ADBUFF *cur, *prev, *first, *p;        /* 入力バッファへのポインタ */
  short int endsignal[ENDSIGNALLEN];     /* 終了信号                 */
  int i_seg;                             /* セグメントindex          */
  int n_seg;                             /* 読み込んだセグメント数   */
  int i_sample;                          /* サンプルindex            */
  float ste;                             /* 短時間エネルギー [dB}    */
  float pth;                             /* 短時間エネルギー閾値     */
  int n_consecutive_seg_belowTH; /* 閾値以下の連続するセグメントの数 */
  int vad_status;                        /* 音声区間検出状態         */
  char name[MAXFNAMELEN];                /* 出力ファイル名接頭辞     */
  int n_utterance;                     /* 発話数(検出した音声区間数) */
  int file_mode;                         /* ファイルモード           */
  char outfile[MAXFNAMELEN];             /* 出力ファイル名           */
  FILE *fp_outfile;                      /* 出力ファイル             */


  /*-------------------------------------
   コマンドラインオプションの処理
  -------------------------------------*/
  strcpy(progname,argv[0]);
  if(3 > argc) {
    usage(progname);
    exit(EXIT_FAILURE);
  }
  file_mode = WRITE_TO_STDOUT;
  while (--argc){
    if(*(s = *++argv) == '-') {
      c = *++s;
      switch(c) {
      case 'a':
	file_mode = WRITE_TO_FILE;
	if(argc > 1) {
	  strcpy(name,*++argv); 
	  --argc;
	}
	else strcpy(name,"x");
	break;
      case 'p':
	pth = atof(*++argv);
	--argc;
	break;
      case 'h':
	usage(progname);
	exit(EXIT_FAILURE);
      default:
	fprintf(stderr,"%s: 無効なオプション '%c'\n",progname,c);
	usage(progname);
	exit(EXIT_FAILURE);
      }
    }
  }


  /*-------------------------------------
    入力音声データの循環バッファを生成
  -------------------------------------*/
  prev = NULL;
  /* 必要な数(BUFFSIZE)のバッファを生成しリンクで繋ぐ */
  for(i_seg=0; i_seg<BUFFSIZE; i_seg++) {
    if(NULL==(cur=(ADBUFF*)calloc(1,sizeof(ADBUFF)))) {
      fprintf(stderr,"%s: error in allocating adbuff\n",progname);
      exit(EXIT_FAILURE);
    }
    if(NULL==(cur->seg=(short int*)calloc(NSEGSAMPLE,sizeof(short int)))) {
      fprintf(stderr,"%s: error in allocating adbuff->samples\n",progname);
      exit(EXIT_FAILURE);
    }
    cur->next = prev;
    prev = cur;
  }
  /* 最後に生成したバッファと最初に生成したバッファを
                  リンクで繋ぎ,循環バッファを形成する */
  first = cur;
  for(i_seg=0; i_seg<BUFFSIZE-1; i_seg++) cur = cur->next;
  cur->next = first;
  

  /*-------------------------------------
    音声区間検出終了信号を生成
  -------------------------------------*/
  for(i_sample=0; i_sample<ENDSIGNALLEN; i_sample++)
    endsignal[i_sample] = ENDSIGNAL;


  /*-----------------------------------------------------------
   標準入力からの信号を一定幅に区切ってセグメントとする。
   閾値を超えるセグメントを音声区間のセグメントとして出力する。
   音声区間中で閾値を下回るセグメントについては,その数が
   連続して5個以下であれば,音声区間とみなす。連続して5個を
   超えたとき,音声区間が終了したと判断し,出力を打ち切る。
  ------------------------------------------------------------*/
  n_utterance = 0;
  //for(;;) { /* 無限ループ */
    n_seg = 0;
    cur = first;
    vad_status = NOTINSPEECH;
    n_consecutive_seg_belowTH = 0;
    while(NSEGSAMPLE == fread(cur->seg,sizeof(short int),NSEGSAMPLE,stdin)) {
      n_seg++;
      if(BUFFSIZE <= n_seg ) { /* バッファにデータが詰まっている状態で処理 */
	ste = 10*log10(short_time_energy(cur->seg,NSEGSAMPLE));
	if(pth <= ste) {    /* 指定した閾値以上のセグメントを音声区間とする */
	  if(NOTINSPEECH == vad_status) {
	    if(file_mode) {
	      /* 検出した音声区間を保存するファイルを開く */
	      n_utterance++;
	      sprintf(outfile,"%s%04d.ad",name,n_utterance);
	      fprintf(stderr,"outfile= <%s>\n",outfile);
	      if(NULL == (fp_outfile = fopen(outfile,"wb"))) {
		fprintf(stderr,"%s: error in opening %s\n",progname,outfile);
		exit(EXIT_FAILURE);
	      }
	    }
	    else {
	      fp_outfile = stdout;
	    }
	    /* バッファに溜っている直前の(BUFFSIZE-1)セグメントを出力 */
	    cur = cur->next;
	    for(i_seg=0; i_seg<BUFFSIZE-1; i_seg++) {
	      if(NSEGSAMPLE != fwrite(cur->seg,sizeof(short int),NSEGSAMPLE,fp_outfile)) {
		fprintf(stderr,"%s: error in writing seg\n",progname);
		exit(EXIT_FAILURE);
	      }
	      cur = cur->next;
	    }
	    vad_status = INSPEECH;
	    n_consecutive_seg_belowTH = 0;
	  }
	  /* 現在のセグメントを出力 */
	  if(NSEGSAMPLE != fwrite(cur->seg,sizeof(short int),NSEGSAMPLE,fp_outfile)) {
	    fprintf(stderr,"%s: error in writing seg\n",progname);
	    exit(EXIT_FAILURE);
	  }
	  fflush(fp_outfile);
	}
	else {
	  if((INSPEECH == vad_status)&&(MAXLOWSEG >= n_consecutive_seg_belowTH)) {
	    n_consecutive_seg_belowTH++;
	    if(NSEGSAMPLE != fwrite(cur->seg,sizeof(short int),NSEGSAMPLE,fp_outfile)) {
	      fprintf(stderr,"%s: error in writing seg\n",progname);
	      exit(EXIT_FAILURE);
	    }
	  }
	  if((INSPEECH == vad_status)&&(MAXLOWSEG < n_consecutive_seg_belowTH)) {
	    if(! file_mode) {  /* ストリーム出力の場合は終了信号を書き出す */
	      if(NSEGSAMPLE != fwrite(endsignal,sizeof(short int),ENDSIGNALLEN,fp_outfile)) {
		fprintf(stderr,"%s: error in writing seg\n",progname);
		exit(EXIT_FAILURE);
	      }
	    }
	    fflush(fp_outfile);
	    n_consecutive_seg_belowTH = 0;
	    break;                         /* データの切り出し終了 */
	  }
	}
      }
      cur = cur->next;
    }
    //}
    return 0;
}





/*---------------------------------------------------------------------

 short_time_energy

---------------------------------------------------------------------*/
float short_time_energy(short int *x,
			int n)
{
  float sum;
  float x_scaled;
  int i;


  sum = 0;
  for(i=0; i<n; i++) {
    x_scaled = (float)x[i]/(float)(-SHRT_MIN);
    sum += x_scaled*x_scaled;
  }

  return sum/n;
}
