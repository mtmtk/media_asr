/**********************************************************************

  vu: 入力音声のパワーの表示
         標準入力からの信号を一定幅に区切ってセグメントとする。
         各セグメントの短時間対数パワーを計算し,
         現在の値,過去の最小値および最大値を表示する。

  使用法
    vu [-h] < file.ad 

  入力
                -h: 使用法を表示
           file.ad: 入力信号波形データファイルあるいはストリーム
                    サンプリング周波数は16kHz, 振幅値は16bit符合付整数
                     

                                              2013年10月19日 高木一幸

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
#define NSEGSAMPLE   (SFREQ*SEGDUR) /* 1セングメントの音声サンプル数 */
#define DB_MIN       -60                 /* 表示パワー最小値[dB]     */
#define DB_MAX       0                   /* 表示パワー最大値[dB]     */
#define DB_STEP      1.25                /* 棒グラフ1文字分のdB値    */




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
  fprintf(stderr, " %s - 入力音声のパワーの表示\n",progname);
  fprintf(stderr, "\n");
  fprintf(stderr, "  使用法:\n");
  fprintf(stderr, "       %s [-h] < file.ad\n",progname);
  fprintf(stderr, "  入力:\n");
  fprintf(stderr, "            -h: 使用法を表示\n");
  fprintf(stderr, "       file.ad: 入力信号波形データファイルあるいはストリーム\n");
  fprintf(stderr, "                サンプリング周波数は16kHz, 振幅値は16bit符号付整数\n");
  fprintf(stderr, "\n");
}





/*---------------------------------------------------------------------

 メインプログラム

---------------------------------------------------------------------*/
int main(int argc, char *argv[]) {
  char progname[MAXFNAMELEN];          /* このプログラムのコマンド名 */
  char *s, c;                  /* コマンドラインオプション処理用変数 */
  short int adbuff[NSEGSAMPLE];        /* 入力音声セグメント       */
  double ste;                          /* 短時間エネルギー [dB}    */
  double ste_sum;                      /* 短時間エネルギー累積値   */
  int n_seg;                           /* セグメント累計数         */
  double ste_max, ste_min, ste_mean;   /* 同上最大値,最小値,平均値 */
  int level_ub;                        /* 棒グラフ表示上限文字数   */
  int level_max;                     /* 棒グラフの暫定最大表示文字数 */
  int level;                           /* 棒グラフの文字数         */
  int l;                               /* 棒グラフの文字index      */


  /*-------------------------------------
   コマンドラインオプションの処理
  -------------------------------------*/
  strcpy(progname,argv[0]);
  if(1 < argc) {
    usage(progname);
    exit(EXIT_FAILURE);
  }


  /*-----------------------------------------------------------
    標準入力からの信号を一定幅に区切ってセグメントとする。
    各セグメントの短時間対数パワーを計算し,
    現在の値,過去の最小値および最大値を表示する。
  ------------------------------------------------------------*/
  /* 棒グラフの目盛を表示 */
  printf("\n Short Time Speech Power\n");
  printf("-60    -50      -40     -30     -20     -10      0 dB\n");
  printf(" +-------+-------+-------+-------+-------+-------+\n");
  level_ub = (DB_MAX-DB_MIN)/DB_STEP;

  /* 音声パワー計算&リアルタイム表示 */
  level_max = -1;
  n_seg = 0;
  ste_sum = 0;
  ste_max = -200;
  ste_min = 200;
  for(;;) { /* 無限ループ */
    while(NSEGSAMPLE == fread(adbuff,sizeof(short int),NSEGSAMPLE,stdin)) {
      n_seg++;
      ste = 10*log10(short_time_energy(adbuff,NSEGSAMPLE));
      ste_sum += ste;
      ste_max = (ste>ste_max)?ste:ste_max;
      ste_min = (ste<ste_min)?ste:ste_min;
      ste_mean = ste_sum/(double)n_seg;
      level = (int)((ste-DB_MIN)/(double)DB_STEP+0.5);
      if(level<0) level = 1;
      level_max = (level>level_max)?level:level_max;
      printf("."); 
      for(l=1; l<=level; l++) printf("=");
      for(l=level+1; l<level_max; l++) printf(" ");
      printf("*");
      for(l=level_max+1; l<=level_ub; l++) printf(" ");
      printf("\r"); 
      fflush(stdout);
    }
  }
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
