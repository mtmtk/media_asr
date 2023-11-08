/**********************************************************************

  ad2fb: フィルタバンク分析
         −音声波形からフィルタバンク出力を計算する−
                     
                                              2013年10月09日 高木一幸
                                              2013年10月13日 高木一幸
                       コメント,変数名を改訂  2013年11月26日 高木一幸

**********************************************************************/
/*---------------------------------------------------------------------

  include files

---------------------------------------------------------------------*/
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>





/*---------------------------------------------------------------------

  macros

---------------------------------------------------------------------*/
#define PREEMCOEF    0.97                /* 高域強調係数             */
#define NYQUISTFREQ  8000                /* ナイキスト周波数 [Hz]    */
#define WINDOW_WIDTH 32                  /* 窓幅 [ms]                */   
#define WINDOW_SHIFT 10                  /* 窓シフト [ms]            */
#define N_DFT        512                 /* FFT点数                  */
#define N_FREQ       256                 /* スペクトル点数           */
#define NUMCHANS     28                  /* フィルタバンクチャネル数 */
#define SWAP(a,b)    tempr=(a);(a)=(b);(b)=tempr /* aとbの値の交換   */





/*---------------------------------------------------------------------

  function prototypes

---------------------------------------------------------------------*/
void ad2fb(int n_frame,                  /* フレーム数               */
	   int n_shift,                  /* シフト幅 [サンプル値]    */
	   short int *ad,                /* 音声サンプル(整数値)     */
	   int n_sample,                 /* 音声サンプル数           */
	   float **fb,                   /* フィルタバンク出力値     */
	   char *progname);              /* プログラム名             */

void preEmphasis(float *x,               /* 音声波形                 */
		 int N);                 /* サンプル数               */

void Hamming(float *w,                   /* 窓関数(重みの配列)       */
	     int N);                     /* 窓幅(サンプル数)         */

void fft(float *x,                       /* 複素数信号               */
	 int nn);                        /* サンプル数               */

void amplitudeSpectrum(float *x,         /* Fourier係数の実部        */
		       float *a,         /* 振幅スペクトル           */
		       int N);
float mel(float f);                      /* 周波数 [Hz]              */

float lnr(float m);                      /* メル周波数 [Mel]         */

int mel2bin(float m);                    /* メル周波数 [Mel]         */





/*---------------------------------------------------------------------

  ad2fb: 音声波形からフィルタバンク出力を計算する

---------------------------------------------------------------------*/
void ad2fb(int n_frame,                  /* フレーム数               */
	   int n_shift,                  /* シフト幅 [サンプル値]    */
	   short int *ad,          /* 窓掛けされた音声サンプル(実数) */
	   int n_sample,                 /* 音声サンプル数           */
	   float **m,                    /* フィルタバンク出力値     */
	   char *progname                /* プログラム名             */
)
{
  int i_frame;                           /* フレームindex            */ 
  int i_sample;                          /* 音声サンプルindex        */
  int ist;                               /* 分析開始サンプル番号     */
  float *xs;                             /* 音声サンプル(実数)       */
  float w[N_DFT];                        /* 窓関数                   */
  float xw[N_DFT*2];             /* 窓掛けされた音声サンプル(複素数) */
  float X[N_FREQ];                      /* 振幅スペクトル           */
  int i, j, k;
  float fc[NUMCHANS];                   /* チャンネル中心周波数[Mel] */
  float flo[NUMCHANS];                  /* チャンネル下限周波数[Mel] */
  float fhi[NUMCHANS];                  /* チャンネル上限周波数[Mel] */
  int kc[NUMCHANS];                     /* チャンネル中心周波数[Mel] */
  int klo[NUMCHANS];                    /* チャンネル下限周波数[Mel] */
  int khi[NUMCHANS];                    /* チャンネル上限周波数[Mel] */
  float W_kj;                           /* フィルタ重み              */
  float Aj;                             /* フィルタ重みの2乗和       */


  /*-----------------------------------------
    メルフィルタバンクの各フィルタについて
    最低周波数,中心周波数,最高周波数を計算
  -----------------------------------------*/ 
  flo[0] = 0;
  fhi[NUMCHANS-1] = mel(NYQUISTFREQ);
  for(j=1; j<=NUMCHANS; j++)
    fc[j-1] = (float)j*mel(NYQUISTFREQ)/(float)(NUMCHANS+1);
  for(j=1; j<NUMCHANS; j++)
    flo[j] = fc[j-1];
  for(j=0; j<NUMCHANS-1; j++)
     fhi[j] = fc[j+1];
   
  for(j=0; j<NUMCHANS; j++) {
    kc[j] = mel2bin(fc[j]);
    klo[j] = mel2bin(flo[j]);
    khi[j] = mel2bin(fhi[j]);
  }

  /*-------------------------------------
    音声データを格納する配列を生成
  -------------------------------------*/
  if(NULL==(xs=(float*)calloc(n_sample,sizeof(float)))) {
    fprintf(stderr,"%s: error in allocating xs\n",progname);
    exit(EXIT_FAILURE);
  }

  /*------------------------------------
    音声サンプル(整数)を実数に変換し
    高域強調処理を施す
  ------------------------------------*/
  /* 符号付2バイト整数の最大値で割る */
  for(i_sample=0; i_sample<n_sample; i_sample++) {
    xs[i_sample] = (float)ad[i_sample]/(float)(-SHRT_MIN);
  }
  /* 高域強調 */
  preEmphasis(xs,i_sample);

  /*-------------------------------------
    ハミング窓を生成
  -------------------------------------*/
  Hamming(w,N_DFT);

  /*-------------------------------------
    フレーム分析
  -------------------------------------*/
  for(i_frame=0; i_frame<n_frame; i_frame++) {
    ist = i_frame*n_shift;

    /* 1フレーム分の音声を切り取りハミング窓を掛け,
       複素数の配列(実数部+虚数部)に設定する */
    for(i=0; i<N_DFT; i++) {
      xw[2*i] = w[i]*xs[ist+i];
      xw[2*i+1] = 0;
    }

    /* 離散フーリエ変換 */
    fft(xw-1,N_DFT);

    /* 振幅スペクトルの計算 */
    amplitudeSpectrum(xw,X,N_FREQ);
    
    /*-------------------------------------------------------

      フィルタバンク出力の計算

      音声の各フレームの振幅スペクトルMFCCを計算する。
      したがって,振幅スペクトル X と フィルタバンク出力値 m は
      (フレーム x 次元)の 2次元配列に保存される。

      したがって, (3.1)式,(3.2)式,(3.3)式と
      この関数の変数の関係は次のようになる。
      ----------------- -------------------
      (3.1)(3.2)(3.3)式 この関数  
      ----------------- -------------------
      i                 i
      j                 j
      k                 k
      m_j               m[i_frame][j]
      W_{k,j}           W_kj
      |X_k|             X[k]
      K_{lo}(j)         klo[j]
      K_{c}(j)          kc[j]
      K_{hi}(j)         khi[j]
      Aj                Aj      
      ----------------- -------------------
      添字 i_frame は(3.1)(3.2)式に関係ないが,上記の対応となるので注意。

      なお, たとえば s = Σ・・・の計算を行う場合は、
　    あらかじめ「s = 0;」という代入文で
      変数sの値をリセットしておく必要があることに注意。

    -------------------------------------------------------*/
    for(j=0; j<NUMCHANS; j++) {
      m[i_frame][j] = 0;
      Aj = 0;
      for(k=klo[j]; k<kc[j]; k++) {
	      W_kj = (float)(k - klo[j]) / (float)(kc[j] - klo[j]) /* fill in blank */;
	      m[i_frame][j] += W_kj * X[k] * X[k] /* fill in blank */;
	      Aj += W_kj /* fill in blank */;
      }
      for(k=kc[j]; k<=khi[j]; k++) {
	      W_kj = (float)(khi[j] - k) / (float)(khi[j] - kc[j]) /* fill in blank */;
	      m[i_frame][j] += W_kj * X[k] * X[k] /* fill in blank */;
	      Aj += W_kj /* fill in blank */;
      }
      /* フィルタの面積が等しくなるように正規化 */
      m[i_frame][j] /= Aj;
    }
  }

  //  free(X);                 /* 音声サンプルを格納する配列の領域を開放 */

  return;
}





/*---------------------------------------------------------------------

  preEmphasis: 高域強調

---------------------------------------------------------------------*/
void preEmphasis(float *x,               /* 音声波形                 */
		 int N)                  /* サンプル数               */
{
  int n;


  for(n=N-1; n>=1; n--)
    x[n] -= PREEMCOEF*x[n-1];
  x[0] *= 1.0-PREEMCOEF; 

  return;
}





/*---------------------------------------------------------------------

 Hamming: ハミング窓を生成

---------------------------------------------------------------------*/
void Hamming(float *w,                   /* 窓関数(重みの配列)       */
	     int N)                      /* 窓幅(サンプル数)         */
{
  int n;
  double a;


  a = 2.0*M_PI/(double)(N-1);
  for(n=0; n<N; n++) 
    w[n] = 0.54-0.46*cos(a*(double)n);

  return;
}





/*---------------------------------------------------------------------

 fft: 高速フーリエ変換

      x[]をそのフーリエ変換で置き換える．x[]は長さnnの複素数の配列で
      あるが,x[1..2*nn]に実部と虚部を交互に入れる．nnは2の整数乗で
      なければならない．フーリエ変換後x[]には,nn個の周波数の値について
      複素フーリエ係数の実部と虚部が交互に入る．


      参考文献: W.H.Press, et al., 
                Numerical recipes: the art of scientific computing,
		Third Edition, pp.612--613,
		Cambridge University Press, New York, 2007.

---------------------------------------------------------------------*/
void fft(float *x,
	   int nn)
{
  int n, mmax, m, j, istep, i;
  double wtemp, wr, wpr, wpi, wi, theta;
  double tempr, tempi;


  n = nn<<1;
  j = 1;
  for(i=1; i<n; i+=2) {
    if(j>i) {
      SWAP(x[j],x[i]);
      SWAP(x[j+1],x[i+1]);
    }
    m = n>>1;
    while(m>=2 && j>m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }
  mmax = 2;
  while(n>mmax) {
    istep = mmax<<1;
    theta = 2.0*M_PI/(double)mmax;
    wtemp = sin(0.5*theta);
    wpr = -2.0*wtemp*wtemp;
    wpi = sin(theta);
    wr = 1.0;
    wi = 0.0;
    for(m=1; m<mmax; m+=2) {
      for(i=m; i<=n; i+=istep) {
	j = i+mmax;
	tempr = wr*x[j]-wi*x[j+1];
	tempi = wr*x[j+1]+wi*x[j];
	x[j] = x[i]-tempr;
	x[j+1] = x[i+1]-tempi;
	x[i] += tempr;
	x[i+1] += tempi;
      }
      wr = (wtemp=wr)*wpr-wi*wpi+wr;
      wi = wi*wpr+wtemp*wpi+wi;
    }
    mmax = istep;
  }

  return;
}





/*---------------------------------------------------------------------

 amplitudeSpectrum: 振幅スペクトルの計算

---------------------------------------------------------------------*/
void amplitudeSpectrum(float *xw,        /* Fourier係数(複素数)      */
		       float *a,         /* 振幅スペクトル           */
		       int N)
{
  int i;
  double xr, xi;                      /* Fourier係数の実数部と虚数部 */


  for(i=0; i<N; i++) {
    xr = xw[2*i];
    xi = xw[2*i+1];
    a[i] = sqrt(xr*xr+xi*xi);
  }

  return;
}





/*----------------------------------------x-----------------------------

 mel: メル周波数の計算

---------------------------------------------------------------------*/
float mel(float f)                       /* 周波数 [Hz]              */
{
  return 2595.0*log10(1.0+f/700.0);
}





/*---------------------------------------------------------------------

 lnr: メル周波数を周波数に変換

---------------------------------------------------------------------*/
float lnr(float m)                       /* メル周波数 [Mel]         */
{
  return 700.0*(pow(10.0,m/2595.0)-1);
}





/*---------------------------------------------------------------------

 mel2bin: メル周波数に対応する周波数ビンを計算

---------------------------------------------------------------------*/
int mel2bin(float m)                     /* メル周波数 [Mel]         */
{
  return (int)((lnr(m)/NYQUISTFREQ*N_FREQ)+0.5);
}
