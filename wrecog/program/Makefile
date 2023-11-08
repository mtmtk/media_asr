#***************************************************************
#
# メディア情報学実験「音声認識」Makefile
#
# 作成履歴
#   初版                                2013年11月04日 高木一幸
#   第2版                               2014年08月31日 高木一幸
#   第3版                               2018年08月06日 高木一幸
#   第4版 多次元ガウス密度関数を独立       2023年10月10日 高木一幸
#
#***************************************************************
all: fb mfccf mfcc drillG recogf _recog train prtmfcc

fb: fb.o ad2fb.o
	cc -o fb fb.o ad2fb.o -lm

fb.o: fb.c
	cc -c -o fb.o fb.c

ad2fb.o: ad2fb.c
	cc -c -o ad2fb.o ad2fb.c

mfccf: mfccf.o ad2mfcc.o ad2fb.o fb2mfcc.o
	cc -o mfccf mfccf.o  ad2mfcc.o ad2fb.o fb2mfcc.o -lm

mfccf.o: mfccf.c
	cc -c -o mfccf.o mfccf.c

fb2mfcc.o: fb2mfcc.c
	cc -c -o fb2mfcc.o fb2mfcc.c

ad2mfcc.o: ad2mfcc.c
	cc -c -o ad2mfcc.o ad2mfcc.c

mfcc: mfcc.o ad2mfcc.o ad2fb.o fb2mfcc.o
	cc -o mfcc mfcc.o ad2mfcc.o ad2fb.o fb2mfcc.o -lm

mfcc.o: mfcc.c
	cc -c -o mfcc.o mfcc.c

drillG: drillG.o vtb.o gpdf.o
	cc -o drillG drillG.o vtb.o gpdf.o -lm

drillG.o: drillG.c
	cc -c -o drillG.o drillG.c

recogf: recogf.o vtb.o gpdf.o
	cc -o recogf recogf.o vtb.o gpdf.o -lm

recogf.o: recogf.c
	cc -c -o recogf.o recogf.c

_recog: _recog.o vtb.o gpdf.o
	cc -o _recog _recog.o vtb.o gpdf.o -lm

_recog.o: _recog.c
	cc -c -o _recog.o _recog.c

vtb.o: vtb.c
	cc -c -o vtb.o vtb.c

gpdf.o: gpdf.c
	cc -c -o gpdf.o gpdf.c

train: train.o gpdf.o
	cc -o train train.o gpdf.o -lm

train.o: train.c
	cc -c -o train.o train.c

prtmfcc: prtmfcc.c
	cc -o prtmfcc prtmfcc.c -lm

clean:
	rm *.o
