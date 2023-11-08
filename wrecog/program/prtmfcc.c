/**********************************************************************

  prtmfcc: MFCCファイルの内容をプリントする


  使用法
    % prtmfcc file.mfcc

  入力
      file.mfcc: MFCCファイル

                     
                                               2013年10月13日 高木一幸

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
#define WINDOW_SHIFT 10                  /* 窓シフト [ms]            */




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

 usage: 使用方法

---------------------------------------------------------------------*/
void usage(char *progname) {
  fprintf(stderr, "\n");
  fprintf(stderr, " %s - MFCCファイルの内容をプリントする\n",progname);
  fprintf(stderr, "\n");
  fprintf(stderr, "  使用法:\n");
  fprintf(stderr, "      %s file.mfcc\n",progname);
  fprintf(stderr, "  入力\n");
  fprintf(stderr, "      file.fb: MFCCファイル\n\n");
 }





/*---------------------------------------------------------------------

 メインプログラム

---------------------------------------------------------------------*/
int main(int argc, char *argv[]) {
  char progname[MAXFNAMELEN];          /* このプログラムのコマンド名 */
  char fname_mfcc[MAXFNAMELEN];          /* MFCCファイル名           */
  FILE *fp_mfcc;                         /* MFCCファイル             */
  int n_dim;                             /* MFCC次元数               */
  int i_dim;                             /* MFCC次元index            */
  int n_frame;                           /* フレーム数               */
  int i_frame;                           /* フレームindex            */
  HTKHEADER htk;                         /* HTK形式ヘッダ            */
  float **mfcc;                          /* MFCC                     */
  float t;                               /* フレーム中心時間[ms]     */



  /*-------------------------------------
    コマンドラインオプションの処理
  -------------------------------------*/
  strcpy(progname,argv[0]);
  if(2 > argc ) {
    usage(progname);
    exit(EXIT_FAILURE);
  }
  strcpy(fname_mfcc,argv[1]); 

  /*-------------------------------------
    MFCCファイルを開く
  -------------------------------------*/
  if(NULL== (fp_mfcc = fopen(fname_mfcc,"r"))) {
    printf("%s: error in opening %s\n",progname,fname_mfcc);
    exit(EXIT_FAILURE);
  }
  /* HTK仕様ヘッダーの読み込み */
  if(1 != fread(&htk,sizeof(HTKHEADER),1,fp_mfcc)) {
    fprintf(stderr,"%s: error in reading HTK header from %s\n",progname,fname_mfcc);
    exit(EXIT_FAILURE);
  }
  n_frame = htk.nSamples;
  n_dim = htk.sampSize/sizeof(float);

    
  /* MFCCを格納する配列を生成 */
  if(NULL==(mfcc=(float**)calloc(n_frame,sizeof(float*)))) {
    fprintf(stderr,"%s: error in allocating mfcc\n",progname);
    exit(EXIT_FAILURE);
  }
  for(i_frame=0; i_frame<n_frame; i_frame++) {
    if(NULL==(mfcc[i_frame]=(float*)calloc(n_dim,sizeof(float)))) {
      fprintf(stderr,"%s: error in allocating mfcc[%d]\n",progname,i_frame);
      exit(EXIT_FAILURE);
    }
  }
  /* 特徴量ベクトルの読み込み */
  for(i_frame=0; i_frame<n_frame; i_frame++) {
    if(n_dim != fread(mfcc[i_frame],sizeof(float),n_dim,fp_mfcc)) {
      fprintf(stderr,"%s: error in reading MFCC of frame %d from %s\n",progname,i_frame,fname_mfcc);
      exit(EXIT_FAILURE);
    }
  }
 
  /* MFCCの値を標準出力にプリント */
  for(i_frame=0; i_frame<n_frame; i_frame++) {
    t = (i_frame+1)*WINDOW_SHIFT;
    for(i_dim=0; i_dim<n_dim; i_dim++) {
      printf("%6.1f\t%d\t%f\n",t,(i_dim+1),mfcc[i_frame][i_dim]);
    }
    printf("\n");
  }
  
  fclose(fp_mfcc);


  /* メモリを開放 */
  for(i_frame=0; i_frame<n_frame; i_frame++) free(mfcc[i_frame]);
  free(mfcc);

  exit(EXIT_SUCCESS);
}
